from inspect import formatannotationrelativeto
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import sys
import warnings
import scalogram
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn import svm 

warnings.filterwarnings('ignore')

#FUNCTIONS

featuresNames=[]

def waitforEnter(fstop=False):
    if fstop:
        if sys.version_info[0] == 2:
            raw_input("Press ENTER to continue.")
        else:
            input("Press ENTER to continue.")


#function to create plots for each class
def plotClasses(data1,name1,data2,name2,data3,name3,data4,name4,data5,name5):
    plt.subplot(5,1,1)
    plt.plot(data1)
    plt.title(name1)
    plt.subplot(5,1,2)
    plt.plot(data2)
    plt.title(name2)
    plt.subplot(5,1,3)
    plt.plot(data3)
    plt.title(name3)
    plt.subplot(5,1,4)
    plt.plot(data4)
    plt.title(name4)
    plt.subplot(5,1,5)
    plt.plot(data5)
    plt.title(name5)
    plt.show()

def breakTrainTest(data,oWnd=300,trainPerc=0.6):
    nSamp,nCols=data.shape
    nObs=int(nSamp/oWnd)
    data_obs=data[:nObs*oWnd,:].reshape((nObs,oWnd,nCols))
    
    order=np.random.permutation(nObs)
    order=np.arange(nObs)    #Comment out to random split
    
    nTrain=int(nObs*trainPerc)
    
    data_train=data_obs[order[:nTrain],:,:]
    data_test=data_obs[order[nTrain:],:,:]
    
    return(data_train,data_test)


#function to extract features from each type of sample
def extractFeatures(data,Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    oClass=np.ones((nObs,1))*Class
    for i in range(nObs):
        M1=np.mean(data[i,:,:],axis=0) #media
        Var=np.var(data[i,:,:],axis=0) #variancia
        
        faux=np.hstack((M1,Var))
        features.append(faux)
    
    return(np.array(features),oClass)

featuresNames.append("Mean,")
featuresNames.append("Variance,")

#same as before but only for periods of silence
def extratctSilence(data,threshold=256):
    if(data[0]<=threshold):
        s=[1]
    else:
        s=[]
    for i in range(1,len(data)):
        if(data[i-1]>threshold and data[i]<=threshold):
            s.append(1)
        elif (data[i-1]<=threshold and data[i]<=threshold):
            s[-1]+=1

    return(s)
    
def extractFeaturesSilence(data,Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    oClass=np.ones((nObs,1))*Class
    for i in range(nObs):
        silence_features=np.array([])
        for c in range(nCols):
            silence=extratctSilence(data[i,:,c],threshold=0)
            if len(silence)>0:
                silence_features=np.append(silence_features,[np.mean(silence),np.var(silence)])
            else:
                silence_features=np.append(silence_features,[0,0])
        features.append(silence_features)
    return(np.array(features),oClass)

#create plot for features
def plotFeatures(features,oClass,f1index=0,f2index=1):
    nObs,nFea=features.shape
    colors=['b','g','r','y','m']
    for i in range(nObs):
        plt.plot(features[i,f1index],features[i,f2index],'o'+colors[int(oClass[i])])

    plt.show()

def extractFeaturesWavelet(data,scales=[2,4,8,16,32],Class=0):
    features=[]
    nObs,nSamp,nCols=data.shape
    oClass=np.ones((nObs,1))*Class
    for i in range(nObs):
        scalo_features=np.array([])
        for c in range(nCols):
            #fixed scales->fscales
            scalo,fscales=scalogram.scalogramCWT(data[i,:,c],scales)
            scalo_features=np.append(scalo_features,scalo)
            
        features.append(scalo_features)
        
    return(np.array(features),oClass)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Classes={0:'YouTube',1:'Browsing',2:'P2P', 3:'VideoCall', 4: 'Bot'}

## -- extract data from files -- ##
yt=np.loadtxt('2p1hyoutube.txt')
browsing=np.loadtxt('2p1hbrowsing.txt')
p2p=np.loadtxt('2p1hp2p112kb.txt')
vc=np.loadtxt('2p1hclass.txt')
#change bot here
bot=np.loadtxt('2p1hsimplebot.txt')

## -- show plots of each data type -- ##
plt.figure(1)
plotClasses(yt,'YouTube',browsing,'Browsing',p2p,'P2P',vc,'VideoCall',bot,'Bot')

## -- save training and test part of each sample and show graphics of training part: 0(b)-download 1(g)-upload -- ##
yt_train,yt_test=breakTrainTest(yt)
browsing_train,browsing_test=breakTrainTest(browsing)
p2p_train,p2p_test=breakTrainTest(p2p)
vc_train,vc_test=breakTrainTest(vc)
bot_train,bot_test=breakTrainTest(bot)

plt.figure(2)
plt.subplot(5,1,1)
for i in range(7):
    plt.plot(yt_train[i,:,0],'b')
    plt.plot(yt_train[i,:,1],'g')
plt.title('YouTube')
plt.ylabel('Bytes/sec')
plt.subplot(5,1,2)
for i in range(7):
    plt.plot(browsing_train[i,:,0],'b')
    plt.plot(browsing_train[i,:,1],'g')
plt.title('Browsing')
plt.ylabel('Bytes/sec')
plt.subplot(5,1,3)
for i in range(7):
    plt.plot(vc_train[i,:,0],'b')
    plt.plot(vc_train[i,:,1],'g')
plt.title('VC')
plt.ylabel('Bytes/sec')
plt.subplot(5,1,4)
for i in range(7):
    plt.plot(p2p_train[i,:,0],'b')
    plt.plot(p2p_train[i,:,1],'g')
plt.title('P2P')
plt.ylabel('Bytes/sec')
plt.subplot(5,1,5)
for i in range(7):
    plt.plot(bot_train[i,:,0],'b')
    plt.plot(bot_train[i,:,1],'g')
plt.title('Bot')
plt.ylabel('Bytes/sec')
plt.show()

## -- put corresponding features on each type of data -- ##
features_yt,oClass_yt=extractFeatures(yt_train,Class=0)
features_browsing,oClass_browsing=extractFeatures(browsing_train,Class=1)
features_p2p,oClass_p2p=extractFeatures(p2p_train,Class=2)
features_vc,oClass_vc=extractFeatures(vc_train,Class=3)
features_bot,oClass_bot=extractFeatures(bot_train,Class=4)

features=np.vstack((features_yt,features_browsing,features_p2p,features_vc,features_bot))
oClass=np.vstack((oClass_yt,oClass_browsing,oClass_p2p,oClass_vc,oClass_bot))

## -- same as before but only for periods of silence -- ##
features_ytS,oClass_yt=extractFeaturesSilence(yt_train,Class=0)
features_browsingS,oClass_browsing=extractFeaturesSilence(browsing_train,Class=1)
features_p2pS,oClass_p2p=extractFeaturesSilence(p2p_train,Class=2)
features_vcS,oClass_vc=extractFeaturesSilence(vc_train,Class=3)
features_botS,oClass_bot=extractFeaturesSilence(bot_train,Class=4)

featuresS=np.vstack((features_ytS,features_browsingS,features_p2pS,features_vcS,features_botS))
oClass=np.vstack((oClass_yt,oClass_browsing,oClass_p2p,oClass_vc,oClass_bot))
featuresNames.append("Silence,")
featuresNames.append("Wavelets;")

## -- scales -- ##
scales=[2,4,8,16,32,64,128,256]

## -- sets of features: -- ##
#:1 ->  training set for anomaly detection (only youtube, browsing, p2p and videocall- bot is the anomaly)
trainFeatures_yt,oClass_yt=extractFeatures(yt_train,Class=0)
trainFeatures_browsing,oClass_browsing=extractFeatures(browsing_train,Class=1)
trainFeatures_p2p,oClass_p2p=extractFeatures(p2p_train,Class=2)
trainFeatures_vc,oClass_vc=extractFeatures(vc_train,Class=3)
trainFeatures=np.vstack((trainFeatures_yt,trainFeatures_browsing,trainFeatures_p2p,trainFeatures_vc))

trainFeatures_ytS,oClass_yt=extractFeaturesSilence(yt_train,Class=0)
trainFeatures_browsingS,oClass_browsing=extractFeaturesSilence(browsing_train,Class=1)
trainFeatures_p2pS,oClass_p2p=extractFeaturesSilence(p2p_train,Class=2)
trainFeatures_vcS,oClass_vc=extractFeaturesSilence(vc_train,Class=3)
trainFeaturesS=np.vstack((trainFeatures_ytS,trainFeatures_browsingS,trainFeatures_p2pS,trainFeatures_vcS))

trainFeatures_ytW,oClass_yt=extractFeaturesWavelet(yt_train,scales,Class=0)
trainFeatures_browsingW,oClass_browsing=extractFeaturesWavelet(browsing_train,scales,Class=1)
trainFeatures_p2pW,oClass_p2p=extractFeaturesWavelet(p2p_train,scales,Class=3)
trainFeatures_vcW,oClass_vc=extractFeaturesWavelet(vc_train,scales,Class=4)
trainFeaturesW=np.vstack((trainFeatures_ytW,trainFeatures_browsingW,trainFeatures_p2pW,trainFeatures_vcW))

o2trainClass=np.vstack((oClass_yt,oClass_browsing,oClass_p2p,oClass_vc))
i2trainFeatures=np.hstack((trainFeatures,trainFeaturesS,trainFeaturesW))

#:2 -> training set for traffic classification
trainFeatures_yt,oClass_yt=extractFeatures(yt_train,Class=0)
trainFeatures_browsing,oClass_browsing=extractFeatures(browsing_train,Class=1)
#trainFeatures_bot,oClass_bot=extractFeatures(bot_train,Class=2) #bot não é utilizado para treinar os modelos, pois o seu comportamento é desconhecido
trainFeatures_p2p,oClass_p2p=extractFeatures(p2p_train,Class=2)
trainFeatures_vc,oClass_vc=extractFeatures(vc_train,Class=3)
trainFeatures=np.vstack((trainFeatures_yt,trainFeatures_browsing,trainFeatures_p2p,trainFeatures_vc))

trainFeatures_ytS,oClass_yt=extractFeaturesSilence(yt_train,Class=0)
trainFeatures_browsingS,oClass_browsing=extractFeaturesSilence(browsing_train,Class=1)
#trainFeatures_botS,oClass_bot=extractFeaturesSilence(bot_train,Class=2)
trainFeatures_p2pS,oClass_p2p=extractFeaturesSilence(p2p_train,Class=2)
trainFeatures_vcS,oClass_vc=extractFeaturesSilence(vc_train,Class=3)
trainFeaturesS=np.vstack((trainFeatures_ytS,trainFeatures_browsingS,trainFeatures_p2pS,trainFeatures_vcS))

trainFeatures_ytW,oClass_yt=extractFeaturesWavelet(yt_train,scales,Class=0)
trainFeatures_browsingW,oClass_browsing=extractFeaturesWavelet(browsing_train,scales,Class=1)
#trainFeatures_botW,oClass_bot=extractFeaturesWavelet(bot_train,scales,Class=2)
trainFeatures_p2pW,oClass_p2p=extractFeaturesWavelet(p2p_train,scales,Class=2)
trainFeatures_vcW,oClass_vc=extractFeaturesWavelet(vc_train,scales,Class=3)
trainFeaturesW=np.vstack((trainFeatures_ytW,trainFeatures_browsingW,trainFeatures_p2pW,trainFeatures_vcW))

o3trainClass=np.vstack((oClass_yt,oClass_browsing,oClass_p2p,oClass_vc))
i3trainFeatures=np.hstack((trainFeatures,trainFeaturesS,trainFeaturesW))

#:3 -> test set for anomaly 
testFeatures_yt,oClass_yt=extractFeatures(yt_test,Class=0)
testFeatures_browsing,oClass_browsing=extractFeatures(browsing_test,Class=1)
testFeatures_p2p,oClass_p2p=extractFeatures(p2p_test,Class=2)
testFeatures_vc,oClass_vc=extractFeatures(vc_test,Class=3)
testFeatures_bot,oClass_bot=extractFeatures(bot_test,Class=4)
testFeatures=np.vstack((testFeatures_yt,testFeatures_browsing,testFeatures_p2p,testFeatures_vc,testFeatures_bot))

testFeatures_ytS,oClass_yt=extractFeaturesSilence(yt_test,Class=0)
testFeatures_browsingS,oClass_browsing=extractFeaturesSilence(browsing_test,Class=1)
testFeatures_p2pS,oClass_p2p=extractFeaturesSilence(p2p_test,Class=2)
testFeatures_vcS,oClass_vc=extractFeaturesSilence(vc_test,Class=3)
testFeatures_botS,oClass_bot=extractFeaturesSilence(bot_test,Class=4)
testFeaturesS=np.vstack((testFeatures_ytS,testFeatures_browsingS,testFeatures_p2pS,testFeatures_vcS,testFeatures_botS))

testFeatures_ytW,oClass_yt=extractFeaturesWavelet(yt_test,scales,Class=0)
testFeatures_browsingW,oClass_browsing=extractFeaturesWavelet(browsing_test,scales,Class=1)
testFeatures_p2pW,oClass_p2p=extractFeaturesWavelet(p2p_test,scales,Class=2)
testFeatures_vcW,oClass_vc=extractFeaturesWavelet(vc_test,scales,Class=3)
testFeatures_botW,oClass_bot=extractFeaturesWavelet(bot_test,scales,Class=4)
testFeaturesW=np.vstack((testFeatures_ytW,testFeatures_browsingW,testFeatures_p2pW,testFeatures_vcW,testFeatures_botW))

o3testClass=np.vstack((oClass_yt,oClass_browsing,oClass_p2p,oClass_vc,oClass_bot))
i3testFeatures=np.hstack((testFeatures,testFeaturesS,testFeaturesW))

#:4 -> test set for classification
o3testClassClassification=np.vstack((oClass_yt,oClass_browsing,oClass_p2p,oClass_vc))

## -- features normalization (acertar a escala) -- ##
i2trainScaler = MaxAbsScaler().fit(i2trainFeatures)
i2trainFeaturesN=i2trainScaler.transform(i2trainFeatures)

i3trainScaler = MaxAbsScaler().fit(i3trainFeatures)  
i3trainFeaturesN=i3trainScaler.transform(i3trainFeatures)

i3AtestFeaturesN=i2trainScaler.transform(i3testFeatures)
i3CtestFeaturesN=i3trainScaler.transform(i3testFeatures)

## -- reduce previous features to only 3 main components (nao percebi bem) -- ##
pca = PCA(n_components=3, svd_solver='full')

i2trainPCA=pca.fit(i2trainFeaturesN)
i2trainFeaturesNPCA = i2trainPCA.transform(i2trainFeaturesN)

i3trainPCA=pca.fit(i3trainFeaturesN)
i3trainFeaturesNPCA = i3trainPCA.transform(i3trainFeaturesN)

i3AtestFeaturesNPCA = i2trainPCA.transform(i3AtestFeaturesN)
i3CtestFeaturesNPCA = i3trainPCA.transform(i3CtestFeaturesN)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#ANOMALY DETECTION
## -- 14 -- ##

s="Features utilizadas: "
for i in featuresNames:
    s=s+i+" "
print(s+"\n")


print('\n-- Anomaly Detection based on One Class Support Vector Machines (PCA Features) --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesNPCA)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesNPCA)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesNPCA)  

L1=ocsvm.predict(i3AtestFeaturesNPCA)
L2=rbf_ocsvm.predict(i3AtestFeaturesNPCA)
L3=poly_ocsvm.predict(i3AtestFeaturesNPCA)

AnomResults={-1:"Anomaly",1:"OK"}
nSamples=0
failed1=0
failed2=0
failed3=0
nObsTest,nFea=i3AtestFeaturesNPCA.shape
for i in range(nObsTest):
    if nSamples<24: 
        if (AnomResults[L1[i]]) == "Anomaly":
           failed1+=1
        if (AnomResults[L2[i]]) == "Anomaly":
           failed2+=1
        if (AnomResults[L3[i]]) == "Anomaly":
           failed3+=1
    else:
        if (AnomResults[L1[i]]) == "OK":
           failed1+=1
        if (AnomResults[L2[i]]) == "OK":
           failed2+=1
        if (AnomResults[L3[i]]) == "OK":
           failed3+=1
    nSamples+=1
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
percentageL=(nSamples-failed1)*100/nSamples
percentageR=(nSamples-failed2)*100/nSamples
percentageP=(nSamples-failed3)*100/nSamples

print(f'% acerto Kernel Linear: {percentageL}\n% acerto Kernel RBF:{percentageR}\n% acerto Kernel Poly:{percentageP}')
## -- 15 -- ##
from sklearn import svm

print('\n-- Anomaly Detection based on One Class Support Vector Machines --')
ocsvm = svm.OneClassSVM(gamma='scale',kernel='linear').fit(i2trainFeaturesN)  
rbf_ocsvm = svm.OneClassSVM(gamma='scale',kernel='rbf').fit(i2trainFeaturesN)  
poly_ocsvm = svm. OneClassSVM(gamma='scale',kernel='poly',degree=2).fit(i2trainFeaturesN)  

L1=ocsvm.predict(i3AtestFeaturesN)
L2=rbf_ocsvm.predict(i3AtestFeaturesN)
L3=poly_ocsvm.predict(i3AtestFeaturesN)

AnomResults={-1:"Anomaly",1:"OK"}

nObsTest,nFea=i3AtestFeaturesN.shape
nSamples=0
failed1=0
failed2=0
failed3=0
for i in range(nObsTest):
    if nSamples<24: 
        if (AnomResults[L1[i]]) == "Anomaly":
           failed1+=1
        if (AnomResults[L2[i]]) == "Anomaly":
           failed2+=1
        if (AnomResults[L3[i]]) == "Anomaly":
           failed3+=1
    else:
        if (AnomResults[L1[i]]) == "OK":
           failed1+=1
        if (AnomResults[L2[i]]) == "OK":
           failed2+=1
        if (AnomResults[L3[i]]) == "OK":
           failed3+=1
    nSamples+=1
    #print('Obs: {:2} ({:<8}): Kernel Linear->{:<10} | Kernel RBF->{:<10} | Kernel Poly->{:<10}'.format(i,Classes[o3testClass[i][0]],AnomResults[L1[i]],AnomResults[L2[i]],AnomResults[L3[i]]))
percentageL=(nSamples-failed1)*100/nSamples
percentageR=(nSamples-failed2)*100/nSamples
percentageP=(nSamples-failed3)*100/nSamples
print(f'% acerto Kernel Linear: {percentageL}\n% acerto Kernel RBF:{percentageR}\n% acerto Kernel Poly:{percentageP}')