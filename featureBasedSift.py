import math

import cv2
import numpy as np
import os
from dataclasses import dataclass

@dataclass
class ClassficationResult:
    classNames: []
    goodMatchList: []
    success: bool
    resultString: str
    matches: []
    kp1: []
    kp2: []
    img1: object
    accuracy: str





class FeatureBasedClassifier:

    #Initialize detector
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.images = []
        self.classNames = []
        self.kp1 = []
        self.kp2 = []
        self.matches = []
        self.allMatchCount = 0
        self.goodMatchList = []

    def loadImageClasses(self):
        #Loading image classes
        print(os.listdir(os.curdir))
        path = 'FeatureBased/Classes'
        # images = []
        # classNames = []

        classList = os.listdir(path)
        classList.remove(".DS_Store")
        print('Classes detected: ',len(classList))

        for imageClass in classList:
            image = cv2.imread(f'{path}/{imageClass}',0)
            self.images.append(image)
            self.classNames.append(os.path.splitext(imageClass)[0])
        print(self.classNames)

    #create descriptiors for the classes
    def getDescriptors(self,imageList):
        descriptorList=[]
        for image in imageList:
            kp,des = self.sift.detectAndCompute(image,None)
            descriptorList.append(des)
            self.kp1.append(kp)
        return descriptorList

    #classify the image to the class
    def getId(self,image, descriptorList):
        #find descriptor of the image to be classified
        kp,des = self.sift.detectAndCompute(image,None)
        self.kp2.append(kp)
        bruteForceMatcher = cv2.BFMatcher()
        matchList = []
        imageClassifiedToIndex = -1
        minNumberOfMatches = 2
        try:
            for descriptor in descriptorList:
                bruteForceMatches = bruteForceMatcher.knnMatch(descriptor,des, k=2)
                bruteForceMatches2 = bruteForceMatcher.match(descriptor,des)
                goodMatches = []
                for i,j in bruteForceMatches:
                    if i.distance < 0.75 * j.distance:
                        goodMatches.append(i)
                matchList.append(len(goodMatches))
                self.allMatchCount+= len(goodMatches)
                self.matches.append(goodMatches)
        except:
            pass
        if len(matchList) != 0:
            if max(matchList) > minNumberOfMatches:
                imageClassifiedToIndex = matchList.index(max(matchList))
        print(matchList)
        self.goodMatchList.append(matchList)

        return imageClassifiedToIndex

    def getAccuracy(self, id):
        mean = self.allMatchCount / len(self.goodMatchList[0])
        sum = 0
        for matchCount in self.goodMatchList[0]:
            sum += pow((matchCount - mean),2)

        deviation = math.sqrt(sum/self.allMatchCount)

        accuracy = self.goodMatchList[0][id] - mean
        accurayDividedByDeviation = accuracy / deviation

        return accurayDividedByDeviation

    def cleanup(self):
        self.kp1 = []
        self.kp2 = []
        self.matches = []
        self.allMatchCount = 0
        self.goodMatchList = []

    def classify(self, image):
        descriptorList = self.getDescriptors(self.images)
        # print(len(descriptorList))
        id = self.getId(image, descriptorList)

        if id != -1:
            accuracy= str(round(self.getAccuracy(id),2))
            result = ClassficationResult(success=True,
                                         resultString="Result: " + self.classNames[id],
                                         matches=self.matches[id],
                                         kp1=self.kp1[id],
                                         kp2=self.kp2[0],
                                         img1=self.images[id],
                                         accuracy=str(accuracy),
                                         classNames=self.classNames,
                                         goodMatchList=self.goodMatchList
                                         )
            self.cleanup()
            return result
        else:
            result = ClassficationResult(success=False,resultString="Result: Unknown")
            self.cleanup()
            return result

