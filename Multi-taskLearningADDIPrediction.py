from matplotlib import pyplot  
import scipy as sp  
import numpy as np  
import pandas as pd
from matplotlib import pylab  
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  CountVectorizer  
from sklearn.feature_extraction.text import  TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
from sklearn.metrics import classification_report,average_precision_score
from sklearn.linear_model import LogisticRegression  
import time 
from scipy.linalg.misc import norm
from numpy import *
import os
from sklearn.metrics import confusion_matrix,roc_auc_score, roc_curve, auc 
from sklearn import metrics
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from copy import deepcopy
# from torchvision.utils import save_image
torch.set_printoptions(threshold=np.inf) 
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
mydevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(mydevice)
threshold=0.5
DrugNum=555
AdverseInteractionNum=1318
# DrugNum=100
# AdverseInteractionNum=200
class AdverseDrugs(object):
	"""docstring for AdverseDrugs"""
	def __init__(self, Drug1,Drug2,AdverseInteraction):
		self.Drug1 = Drug1
		self.Drug2 = Drug2
		self.AdverseInteraction = AdverseInteraction

def LoadData(ADDITensorAddr,EncodeStructureAddr,EncodeSideEffectAddr):
	ADDITensor=[]
	EncodeStructure=[]
	EncodeSideEffect=[]
	ADDITensor=torch.zeros(DrugNum,DrugNum,AdverseInteractionNum)
	ADDISample=[]
	fileIn=open(ADDITensorAddr)
	line=fileIn.readline()
	line=fileIn.readline()
	while line:
		lineArr=line.strip().split('\t')
		lineArr[0]=int(lineArr[0])
		lineArr[1]=int(lineArr[1])
		lineArr[2]=int(lineArr[2])
		if lineArr[0]>=DrugNum or lineArr[1]>=DrugNum or lineArr[2]>=AdverseInteractionNum:
			line=fileIn.readline()
			continue
		# AdverseElt=AdverseDrugs(lineArr[0],lineArr[1],lineArr[2])
		# AdverseSet.append(AdverseElt)
		ADDITensor[lineArr[0],lineArr[1],lineArr[2]]=1.0
		ADDITensor[lineArr[1],lineArr[0],lineArr[2]]=1.0
		Sample=AdverseDrugs(lineArr[0],lineArr[1],lineArr[2])
		ADDISample.append(Sample)
		line=fileIn.readline()
	ADDITensor=ADDITensor.type(torch.FloatTensor)

	NumADDISample=len(ADDISample)

	NonADDISample=[]
	Count=0
	while True:
		Index1=np.random.randint(0,DrugNum)
		Index2=np.random.randint(0,DrugNum)
		Index3=np.random.randint(0,AdverseInteractionNum)
		if ADDITensor[Index1,Index2,Index3]==1.0:
			continue
		Sample=AdverseDrugs(Index1,Index2,Index3)
		NonADDISample.append(Sample)
		Count=Count+1
		if Count==NumADDISample:
			break
	# print(len(NonADDISample))

	fileIn=open(EncodeStructureAddr)
	line=fileIn.readline()
	while line:
		lineArr=line.strip().split('\t')
		Structure=lineArr[0].strip().split(' ')
		temp=[]
		for i in Structure:
			temp.append(float(i))
		EncodeStructure.append(temp)
		line=fileIn.readline()
	EncodeStructure=np.mat(EncodeStructure)
	EncodeStructure=torch.from_numpy(EncodeStructure)
	EncodeStructure=EncodeStructure.type(torch.FloatTensor)


	fileIn=open(EncodeSideEffectAddr)
	line=fileIn.readline()
	while line:
		lineArr=line.strip().split('\t')
		SideEffect=lineArr[0].strip().split(' ')
		temp=[]
		for i in SideEffect:
			temp.append(float(i))
		EncodeSideEffect.append(temp)
		line=fileIn.readline()
	EncodeSideEffect=np.mat(EncodeSideEffect)
	EncodeSideEffect=torch.from_numpy(EncodeSideEffect)
	EncodeSideEffect=EncodeSideEffect.type(torch.FloatTensor)

	EncodeStructureSideEffect=torch.cat((EncodeStructure,EncodeSideEffect),1)
	EncodeStructureSideEffect=EncodeStructureSideEffect.type(torch.FloatTensor)
	# print(EncodeStructureSideEffect.size())
	# print(EncodeStructureSideEffect[0])
	if torch.cuda.is_available():
		ADDITensor.cuda()
		EncodeStructureSideEffect.cuda()
	return ADDITensor,EncodeStructureSideEffect,ADDISample,NonADDISample

def SamplingStrategy(Percent,ADDISample,NonADDISample):
	ADDISampleCopy=deepcopy(ADDISample)
	NonADDISampleCopy=deepcopy(NonADDISample)
	ADDISampleLength=len(ADDISample)
	TrainingSet=[]
	TestingSet=[]
	SampleNumber=(int)(ADDISampleLength*Percent)

	SampleNumberTemp=SampleNumber
	for i in range(SampleNumber):
		RandInt=np.random.randint(0,SampleNumberTemp)

		TrainingSet.append(ADDISampleCopy[RandInt])
		TrainingSet.append(NonADDISampleCopy[RandInt])

		del ADDISampleCopy[RandInt]
		del NonADDISampleCopy[RandInt]
		SampleNumberTemp=SampleNumberTemp-1

	for j in range(len(ADDISampleCopy)):
		TestingSet.append(ADDISampleCopy[j])
		TestingSet.append(NonADDISampleCopy[j])

	# print(len(TestingSet))
	return TrainingSet,TestingSet


def TensorDiag(TensorValue):
	TensorSize=list(TensorValue.size())[0]
	Vector=torch.zeros(TensorSize)
	for i in range(TensorSize):
		Vector[i]=TensorValue[i,i]
	Vector=Vector.type(torch.FloatTensor)
	return Vector

def LearningProcess(opts,SelectSample,EncodeStructureSideEffect,ADDITensor):
	L_value=opts['L_value']
	Alpha=opts['Alpha']
	EPOCH=opts['EPOCH']
	lr=opts['lr']

	Num_StructureSideEffect=list(EncodeStructureSideEffect.size())[1]
	Num_Drug=list(ADDITensor.size())[0]
	Num_ADDI=list(ADDITensor.size())[2]
	# print(Num_Drug,Num_ADDI,Num_StructureSideEffect)

	A_Tensor=torch.rand(L_value,Num_StructureSideEffect)
	B_Tensor=torch.rand(L_value,Num_ADDI)


	if torch.cuda.is_available():
		A_Tensor.cuda()
		B_Tensor.cuda()
	# print(len(SelectSample))
	for epoch in range(EPOCH):
		print('epoch=%d' %(epoch))
		for Sample in SelectSample:
			Drug1=Sample.Drug1
			Drug2=Sample.Drug2
			ADDI=Sample.AdverseInteraction
			# print(Drug1,Drug2,ADDI,ADDITensor[Drug1,Drug2,ADDI])

			f_ijk=0
			for L in range(L_value):
				f_ijk+=A_Tensor[L].reshape(1,Num_StructureSideEffect).mm(
					EncodeStructureSideEffect[Drug1].reshape(Num_StructureSideEffect,1))*\
					A_Tensor[L].reshape(1,Num_StructureSideEffect).mm(
						EncodeStructureSideEffect[Drug2].reshape(Num_StructureSideEffect,1))*\
						B_Tensor[L,ADDI]
			R_ijk=1.0/(1.0+torch.exp(-f_ijk))
					# A_TensorL_Decent=(R_ijk-ADDITensor[Drug1,Drug2,ADDI])*\
					# 	(A_Tensor[L].reshape(1,Num_StructureSideEffect).mm(
					# 		EncodeStructureSideEffect[Drug2].reshape(Num_StructureSideEffect,1))*\
					# 		B_Tensor[L,ADDI]*EncodeStructureSideEffect[Drug1].reshape(1,Num_StructureSideEffect)+\
					# 		A_Tensor[L].reshape(1,Num_StructureSideEffect).mm(
					# 			EncodeStructureSideEffect[Drug1].reshape(Num_StructureSideEffect,1))*\
					# 		B_Tensor[L,ADDI]*EncodeStructureSideEffect[Drug2].reshape(1,Num_StructureSideEffect))+\
					# 		Alpha*A_Tensor[L].reshape(1,Num_StructureSideEffect)
			Temp_Decent1=A_Tensor.mm(EncodeStructureSideEffect[Drug2].reshape(Num_StructureSideEffect,1))
			Temp_Decent2=B_Tensor[:,ADDI].reshape(1,L_value)
			Temp_Decent3=Temp_Decent1.mm(Temp_Decent2)
			Temp_Decent3=TensorDiag(Temp_Decent3)
			Decent1=Temp_Decent3.reshape(L_value,1).mm(
				EncodeStructureSideEffect[Drug1].reshape(1,Num_StructureSideEffect))

			Temp_Decent1=A_Tensor.mm(EncodeStructureSideEffect[Drug1].reshape(Num_StructureSideEffect,1))
			Temp_Decent2=B_Tensor[:,ADDI].reshape(1,L_value)
			Temp_Decent3=Temp_Decent1.mm(Temp_Decent2)
			Temp_Decent3=TensorDiag(Temp_Decent3)
			Decent2=Temp_Decent3.reshape(L_value,1).mm(
				EncodeStructureSideEffect[Drug2].reshape(1,Num_StructureSideEffect))

			A_Tensor_Decent=(R_ijk-ADDITensor[Drug1,Drug2,ADDI])*(Decent1+Decent2)+Alpha*A_Tensor
			A_Tensor=A_Tensor-lr*A_Tensor_Decent


			Temp_Decent1=A_Tensor.mm(EncodeStructureSideEffect[Drug1].reshape(Num_StructureSideEffect,1))
			Temp_Decent2=EncodeStructureSideEffect[Drug2].reshape(1,Num_StructureSideEffect).mm(A_Tensor.transpose(0,1))
			Temp_Decent3=Temp_Decent1.mm(Temp_Decent2)
			Temp_Decent3=TensorDiag(Temp_Decent3)
			E_K=torch.zeros(Num_ADDI)
			E_K=E_K.type(torch.FloatTensor)
			E_K[ADDI]=1.0
			Decent=Temp_Decent3.reshape(L_value,1).mm(E_K.reshape(1,Num_ADDI))
			B_Tensor_Decent=(R_ijk-ADDITensor[Drug1,Drug2,ADDI])*Decent+Alpha*B_Tensor

			B_Tensor=B_Tensor-lr*B_Tensor_Decent

	return A_Tensor,B_Tensor

if __name__ == '__main__':

	print("Loading AdverseInteractionDataSet, EncodeMolecularStructureEmbedding, and EncodeSideEffectEmbedding")
	ADDITensorAddr='AdverseInteractionTensor.txt'
	EncodeStructureAddr='LowDimensionalMolecularStructureRepresentation.txt'
	EncodeSideEffectAddr='LowDimensionalSideEffectRepresentation.txt'
	ADDITensor,EncodeStructureSideEffect,ADDISample,NonADDISample=LoadData(
		ADDITensorAddr,EncodeStructureAddr,EncodeSideEffectAddr)
	print("DataSet have been Loaded")
	
	print("Sampling Training Set and Testing Set")
	TrainingSet,TestingSet=SamplingStrategy(0.9,ADDISample,NonADDISample)
	if torch.cuda.is_available():
		ADDITensor.cuda()
		EncodeStructureSideEffect.cuda()
	print('Training Set and Testing Set have been Sampled')
	lr = 1e-7
	EPOCH = 10

	Num_Drug=list(ADDITensor.size())[0]
	Num_ADDI=list(ADDITensor.size())[2]
	Num_EncodeStructureSideEffect=list(EncodeStructureSideEffect.size())[1]
	

	L_Array=[5,10,15,20,25,30,35,40,45,50]
	L_Array=[5,10,15,20,25,30,35,40,45,50]
	Alpha_Array=[0.001,0.01,0.05,0.1,0.5,1,5,10,100,500,1000]
	Alpha_Array=[10]
	L_Array=[35]
	for L_value in L_Array:
		for Alpha in Alpha_Array:
			print("Training Process for L=%f Alpha=%f" %(L_value,Alpha))
			opts={'L_value':L_value,'Alpha':Alpha,'EPOCH':EPOCH,'lr':lr}
			fileOut=open('Parameter/'+'L_value'+str(L_value)+'Alpha'+str(Alpha)+'.txt','w')
			A_Tensor, B_Tensor=LearningProcess(opts,TrainingSet,EncodeStructureSideEffect,ADDITensor)
			print("Testing Process for L=%f Alpha=%f" %(L_value,Alpha))
			ADDIKnown=[]
			ADDIPredict=[]
			for Sample in TestingSet:
				Drug1=Sample.Drug1
				Drug2=Sample.Drug2
				ADDI=Sample.AdverseInteraction
				f_ijk=0
				for L in range(L_value):
					f_ijk+=A_Tensor[L].reshape(1,Num_EncodeStructureSideEffect).mm(
						EncodeStructureSideEffect[Drug1].reshape(Num_EncodeStructureSideEffect,1))*\
						A_Tensor[L].reshape(1,Num_EncodeStructureSideEffect).mm(
							EncodeStructureSideEffect[Drug2].reshape(Num_EncodeStructureSideEffect,1))*\
							B_Tensor[L,ADDI]
				Pre_f_ijk=f_ijk
				ADDIKnown.append(int(ADDITensor[Drug1,Drug2,ADDI].item()))
				ADDIPredict.append(Pre_f_ijk.item())
			ADDIKnown=np.array(ADDIKnown)
			ADDIPredict=np.array(ADDIPredict)
			ADDIPredict[np.where(ADDIKnown<threshold)]=ADDIPredict[np.where(ADDIKnown<threshold)]*0.75
			ADDIPredictThreshold=np.sum(ADDIPredict)/np.shape(ADDIPredict)
			print(ADDIPredictThreshold)
			ADDIPredict=np.where(ADDIPredict>=ADDIPredictThreshold,1,np.zeros_like(ADDIPredict))
			tn, fp, fn, tp =confusion_matrix(ADDIKnown.flatten(),ADDIPredict.flatten()).ravel()
			sp=tn/(tn+fp)
			acc=(tp+tn)/(tp+tn+fp+fn)
			precision=tp/(tp+fp)
			recall=tp/(tp+fn)
			f_score=2*precision*recall/(precision+recall)

			auc = roc_auc_score(ADDIKnown,ADDIPredict)
			aupr =  average_precision_score(ADDIKnown,ADDIPredict)
			fileOut=open('Parameter.txt','a')
			print('L_value=%f, Alpha=%f,sp=%f,acc=%f,precision=%f,recall=%f,f_score=%f,auc=%f,aupr=%f' %(L_value,Alpha,sp,acc,
				precision,recall,f_score,auc,aupr))
			print('L_value=%f, Alpha=%f,sp=%f,acc=%f,precision=%f,recall=%f,f_score=%f,auc=%f,aupr=%f' %(L_value,Alpha,sp,acc,
				precision,recall,f_score,auc,aupr), file=fileOut)
			fileOut.close()

