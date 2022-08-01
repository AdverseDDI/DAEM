from matplotlib import pyplot  
import scipy as sp  
import numpy as np  
import pandas as pd
from matplotlib import pylab  
from sklearn.datasets import load_files  
import sklearn.model_selection
from sklearn.feature_extraction.text import  CountVectorizer  
from sklearn.feature_extraction.text import  TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
from sklearn.metrics import classification_report  
from sklearn.linear_model import LogisticRegression  
import time 
from scipy.linalg.misc import norm
from numpy import *
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc 
from sklearn import metrics
import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
# from torchvision.utils import save_image
torch.set_printoptions(threshold=np.inf) 

class AutoEncoder(nn.Module):
	def __init__(self, Num_Structure, Num_SideEffect):
		super(AutoEncoder, self).__init__()

		Num_Structure_hidden1=(int)(Num_Structure/3)
		Num_Structure_hidden2=(int)(Num_Structure_hidden1/3)
		Num_Structure_hidden3=(int)(Num_Structure_hidden2/3)



		Num_SideEffect_hidden1=(int)(Num_SideEffect/3)
		Num_SideEffect_hidden2=(int)(Num_SideEffect_hidden1/3)
		Num_SideEffect_hidden3=(int)(Num_SideEffect_hidden2/3)


		self.EncoderStructure=nn.Sequential(
			nn.Linear(Num_Structure,Num_Structure),
			nn.Sigmoid(),
			nn.Linear(Num_Structure,Num_Structure_hidden1),
			nn.Sigmoid(),
			nn.Linear(Num_Structure_hidden1,Num_Structure_hidden2),
			nn.Sigmoid(),
			)
		self.DecoderStructure=nn.Sequential(
			nn.Linear(Num_Structure_hidden2,Num_Structure_hidden1),
			nn.Sigmoid(),
			nn.Linear(Num_Structure_hidden1,Num_Structure),
			nn.Sigmoid(),
			nn.Linear(Num_Structure,Num_Structure),
			nn.Sigmoid(),
			)

		self.EncoderSideEffect=nn.Sequential(
			nn.Linear(Num_SideEffect,Num_SideEffect),
			nn.Sigmoid(),
			nn.Linear(Num_SideEffect,Num_SideEffect_hidden1),
			nn.Sigmoid(),
			nn.Linear(Num_SideEffect_hidden1,Num_SideEffect_hidden2),
			nn.Sigmoid(),
			)
		self.DecoderSideEffect=nn.Sequential(
			nn.Linear(Num_SideEffect_hidden2,Num_SideEffect_hidden1),
			nn.Sigmoid(),
			nn.Linear(Num_SideEffect_hidden1,Num_SideEffect),
			nn.Sigmoid(),
			nn.Linear(Num_SideEffect,Num_SideEffect),
			nn.Sigmoid(),
			)

	def forward(self, StructureVect,SideEffectVect):
		EncodeStructure=self.EncoderStructure(StructureVect)
		DecodeStructure=self.DecoderStructure(EncodeStructure)

		EncodeSideEffect=self.EncoderSideEffect(SideEffectVect)
		DecodeSideEffect=self.DecoderSideEffect(EncodeSideEffect)
		return EncodeStructure,DecodeStructure,EncodeSideEffect,DecodeSideEffect

def H_Matrix_Cal(Num_Drug):
	Identity=torch.eye(Num_Drug)
	AllOneMatrix=torch.ones((Num_Drug,Num_Drug))
	AllOneMatrix=1.0*AllOneMatrix/Num_Drug
	H_Matrix=Identity-AllOneMatrix
	return H_Matrix


def LoadData(DrugStructureAddr,DrugSideEffectAddr,ADDINetworkAddr):
	DrugStructureMatrix=[]
	DrugSideEffectMatrix=[]
	ADDINetworkMatrix=[]
	fileIn=open(DrugStructureAddr)
	line=fileIn.readline()
	while line:
		lineArr=line.strip().split('\t')
		lineSpace=lineArr[1].strip().split()
		temp=[]
		for i in lineSpace:
			temp.append(float(i))
		DrugStructureMatrix.append(temp)
		line=fileIn.readline()
	DrugStructureMatrix=np.mat(DrugStructureMatrix)
	DrugStructureTensor=torch.from_numpy(DrugStructureMatrix)


	fileIn=open(DrugSideEffectAddr)
	line=fileIn.readline()
	while line:
		lineArr=line.strip().split('\t')
		lineSpace=lineArr[1].strip().split()
		temp=[]
		for i in lineSpace:
			temp.append(float(i))
		DrugSideEffectMatrix.append(temp)
		line=fileIn.readline()
	DrugSideEffectMatrix=np.mat(DrugSideEffectMatrix)
	DrugSideEffectTensor=torch.from_numpy(DrugSideEffectMatrix)
	
	fileIn=open(ADDINetworkAddr)
	line=fileIn.readline()
	while line:
		lineArr=line.strip().split('\t')
		lineSpace=lineArr[0].strip().split()
		temp=[]
		for i in lineSpace:
			temp.append(int(i))
		ADDINetworkMatrix.append(temp)
		line=fileIn.readline()
	ADDINetworkMatrix=np.mat(ADDINetworkMatrix)
	ADDINetworkTensor=torch.from_numpy(ADDINetworkMatrix)
	return DrugStructureTensor,DrugSideEffectTensor,ADDINetworkTensor


class ObjectiveFunctionLoss(nn.Module):
	def __init__(self):
		super(ObjectiveFunctionLoss, self).__init__()

	def forward(self,Sample_StructureTest,EncodeStructure,
		DecodeStructure,Sample_SideEffectTest,EncodeSideEffect,DecodeSideEffect,ADDINetworkTensor,
		Alpha,Beta):
		StructureEmbeddingNum=np.shape(EncodeStructure)[1]
		SideEffectEmbeddingNum=np.shape(EncodeSideEffect)[1]
		# print(StructureEmbeddingNum,SideEffectEmbeddingNum)
		if torch.cuda.is_available():
			Sample_StructureTest.cuda()
			EncodeStructure.cuda()
			Sample_SideEffectTest.cuda()
			Sample_SideEffectTest.cuda()
			EncodeSideEffect.cuda()
			DecodeSideEffect.cuda()
		criterion = nn.MSELoss()
		Loss1=criterion(Sample_StructureTest,DecodeStructure)
		Loss2=criterion(Sample_SideEffectTest,DecodeSideEffect)
		# (Loss1,Loss2)
		Num_Drug=Sample_StructureTest.size()[0]
		## print(EncodeStructure.transpose(0,1).size())
		
		RowNorm1=torch.norm(EncodeStructure,dim=1).reshape(Num_Drug,-1)
		RowNorm2=torch.norm(EncodeStructure,dim=1).reshape(-1,Num_Drug)
		TempMatrixStructure=torch.mm(RowNorm1,RowNorm2)
		StructureSimilarity=torch.mm(EncodeStructure,EncodeStructure.transpose(0,1))
		StructureSimilarity=torch.div(StructureSimilarity,TempMatrixStructure)


		RowNorm1=torch.norm(EncodeSideEffect,dim=1).reshape(Num_Drug,-1)
		RowNorm2=torch.norm(EncodeSideEffect,dim=1).reshape(-1,Num_Drug)
		TempMatrixSideEffect=torch.mm(RowNorm1,RowNorm2)
		SideEffectSimilarity=torch.mm(EncodeSideEffect,EncodeSideEffect.transpose(0,1))
		SideEffectSimilarity=torch.div(SideEffectSimilarity,TempMatrixSideEffect)


		StructureSimilarity=torch.div(1.0,1.0+torch.exp(-StructureSimilarity))
		SideEffectSimilarity=torch.div(1.0,1.0+torch.exp(-SideEffectSimilarity))
		# print(StructureSimilarity)
		# print(SideEffectSimilarity)

		Loss3=Alpha*((torch.log(1-StructureSimilarity)*ADDINetworkTensor).sum()+\
			(torch.log(SideEffectSimilarity)*ADDINetworkTensor).sum())
		# print(Loss3)
		Loss3=-Loss3

		# print('****%f' %(torch.log(1-StructureSimilarity).mul(ADDINetworkTensor)).sum())
		H_Matrix=H_Matrix_Cal(Num_Drug)
		if torch.cuda.is_available():
			H_Matrix.cuda()
		Loss4=-Beta*torch.trace(EncodeStructure.mm(EncodeStructure.transpose(0,1)).mm(
			H_Matrix).mm(EncodeSideEffect).mm(EncodeSideEffect.transpose(0,1)).mm(
			H_Matrix.transpose(0,1)))

		# print(Loss4)
		# print(Loss3,Loss4)
		Losss=Loss1.item()+Loss2.item()+Loss3.item()+Loss4.item()
		Loss=Loss1+Loss2+Loss3+Loss4
		print(Losss)
		return Loss


# def ObjectiveFunctionLoss(Sample_StructureTest,EncodeStructure,
# 	DecodeStructure,Sample_SideEffectTest,EncodeSideEffect,DecodeSideEffect,ADDINetworkTensor,
# 	Alpha,Beta):
# 	StructureEmbeddingNum=np.shape(EncodeStructure)[1]
# 	SideEffectEmbeddingNum=np.shape(EncodeSideEffect)[1]
# 	# print(StructureEmbeddingNum,SideEffectEmbeddingNum)
# 	if torch.cuda.is_available():
# 		Sample_StructureTest.cuda()
# 		EncodeStructure.cuda()
# 		Sample_SideEffectTest.cuda()
# 		Sample_SideEffectTest.cuda()
# 		EncodeSideEffect.cuda()
# 		DecodeSideEffect.cuda()
# 	criterion = nn.MSELoss()
# 	Loss1=criterion(Sample_StructureTest,DecodeStructure)
# 	Loss2=criterion(Sample_SideEffectTest,DecodeSideEffect)
# 	print(Loss1,Loss2)
# 	Num_Drug=Sample_StructureTest.size()[0]
# 	## print(EncodeStructure.transpose(0,1).size())
	
# 	RowNorm1=torch.norm(EncodeStructure,dim=1).reshape(Num_Drug,-1)
# 	RowNorm2=torch.norm(EncodeStructure,dim=1).reshape(-1,Num_Drug)
# 	TempMatrixStructure=torch.mm(RowNorm1,RowNorm2)
# 	StructureSimilarity=torch.mm(EncodeStructure,EncodeStructure.transpose(0,1))
# 	StructureSimilarity=torch.div(StructureSimilarity,TempMatrixStructure)


# 	RowNorm1=torch.norm(EncodeSideEffect,dim=1).reshape(Num_Drug,-1)
# 	RowNorm2=torch.norm(EncodeSideEffect,dim=1).reshape(-1,Num_Drug)
# 	TempMatrixSideEffect=torch.mm(RowNorm1,RowNorm2)
# 	SideEffectSimilarity=torch.mm(EncodeSideEffect,EncodeSideEffect.transpose(0,1))
# 	SideEffectSimilarity=torch.div(SideEffectSimilarity,TempMatrixSideEffect)


# 	StructureSimilarity=torch.div(1.0,1.0+torch.exp(-StructureSimilarity))
# 	SideEffectSimilarity=torch.div(1.0,1.0+torch.exp(-SideEffectSimilarity))
# 	# print(StructureSimilarity)
# 	# print(SideEffectSimilarity)

# 	Loss3=Alpha*((torch.log(1-StructureSimilarity)*ADDINetworkTensor).sum()+\
# 		(torch.log(SideEffectSimilarity)*ADDINetworkTensor).sum())
# 	# print(Loss3)
# 	Loss3=-Loss3

# 	# print('****%f' %(torch.log(1-StructureSimilarity).mul(ADDINetworkTensor)).sum())
# 	H_Matrix=H_Matrix_Cal(Num_Drug)
# 	if torch.cuda.is_available():
# 		H_Matrix.cuda()
# 	Loss4=-Beta*torch.trace(EncodeStructure.mm(EncodeStructure.transpose(0,1)).mm(
# 		H_Matrix).mm(EncodeSideEffect).mm(EncodeSideEffect.transpose(0,1)).mm(
# 		H_Matrix.transpose(0,1)))

# 	# print(Loss4)
# 	print(Loss3,Loss4)
# 	Losss=Loss1.item()+Loss2.item()+Loss3.item()+Loss4.item()
# 	Loss=Loss1+Loss2+Loss3+Loss4
# 	print(Losss)
# 	return Loss

if __name__ == '__main__':
	
	x=torch.Tensor([1,2,3,4,5])
	y=torch.Tensor([2,3,4,5,6])
	print(torch.div(x,y))

	DrugStructureAddr='MolecularStructureMatrix.txt'
	DrugSideEffectAddr='SiderEffectMatrix.txt'
	ADDINetworkAddr='AdverseInteractionMatrix.txt'
	DrugStructureTensor, DrugSideEffectTensor,ADDINetworkTensor=LoadData(DrugStructureAddr,
		DrugSideEffectAddr,ADDINetworkAddr)

	# print(DrugStructureTensor[0,])
	Num_Drug=list(DrugStructureTensor.size())[0]
	Num_Structure=list(DrugStructureTensor.size())[1]
	Num_SideEffect=list(DrugSideEffectTensor.size())[1]
	# print(Num_Drug,Num_Structure,Num_SideEffect)

	AlphaSet=[0.001,0.01,0.05,0.1,0.501,5,10,100,500,1000]
	BetaSet=[0.001,0.01,0.05,0.1,0.501,5,10,100,500,1000]
	AlphaSet=[100]
	BetaSet=[0.5]

	for Alpha in AlphaSet:
		for Beta in BetaSet:
			lr = 1e-5
			weight_decay =1e-9
			Model=AutoEncoder(Num_Structure,Num_SideEffect)
			LossFunction=ObjectiveFunctionLoss()
			if torch.cuda.is_available():
				Model.cuda()
			optimizier = optim.Adam(Model.parameters(), lr=lr, weight_decay=weight_decay)
			FinalEncodeStructure=0
			FinalEncodeSideEffect=0
			EPOCH = 100
			for epoch in range(EPOCH):

				Sample_StructureTrain=DrugStructureTensor.type(torch.FloatTensor) 
				Sample_StructureTest=DrugStructureTensor.type(torch.FloatTensor)

				Sample_SideEffectTrain=DrugSideEffectTensor.type(torch.FloatTensor)
				Sample_SideEffectTest=DrugSideEffectTensor.type(torch.FloatTensor)

				ADDINetworkTensor=ADDINetworkTensor.type(torch.FloatTensor)
				# print(1-ADDINetworkTensor)
				#print(Sample_x.size())

				EncodeStructure,DecodeStructure,EncodeSideEffect,DecodeSideEffect=Model(
					Sample_StructureTrain,Sample_SideEffectTrain)
				

				Loss=LossFunction(Sample_StructureTest,EncodeStructure,
					DecodeStructure,Sample_SideEffectTest,EncodeSideEffect,DecodeSideEffect,
					ADDINetworkTensor,Alpha,Beta)
				optimizier.zero_grad()	
				Loss.backward(retain_graph=True)
				optimizier.step()
				if epoch==EPOCH-1:
					FinalEncodeStructure=EncodeStructure
					FinalEncodeSideEffect=EncodeSideEffect
					# print(FinalEncodeStructure.size())
					# print(FinalEncodeSideEffect.size())

			fileOut=open('LowDimensionalMolecularStructureRepresentation.txt','w')
			for i in range(Num_Drug):
				for Value in FinalEncodeStructure[i]:
					print('%.4f ' %(Value.data.numpy()),end='',file=fileOut)
				print('',file=fileOut)
			fileOut.close()

			fileOut=open('LowDimensionalSideEffectRepresentation.txt','w')
			for i in range(Num_Drug):
				for Value in FinalEncodeSideEffect[i]:
					print('%.4f ' %(Value.data.numpy()),end='',file=fileOut)
				print('',file=fileOut)
			fileOut.close()


