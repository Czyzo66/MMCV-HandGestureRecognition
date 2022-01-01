import cv2
import numpy as np
import os


class FeatureBasedClassifier:

    #Initialize detector
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.images = []
        self.classNames = []


    def loadImageClasses(self):
        #Loading image classes
        print(os.listdir(os.curdir))
        path = 'Classes'
        # images = []
        # classNames = []

        classList = os.listdir(path)
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
        return descriptorList

    #classify the image to the class
    def getId(self,image, descriptorList):
        #find descriptor of the image to be classified
        kp,des = self.sift.detectAndCompute(image,None)
        bruteForceMatcher = cv2.BFMatcher()
        matchList = []
        imageClassifiedToIndex = -1
        minNumberOfMatches = 2
        try:
            for descriptor in descriptorList:
                bruteForceMatches = bruteForceMatcher.knnMatch(descriptor,des, k=2)
                goodMatches = []
                for i,j in bruteForceMatches:
                    if i.distance < 0.75 * j.distance:
                        goodMatches.append([i])
                matchList.append(len(goodMatches))
        except:
            pass
        if len(matchList) != 0:
            if max(matchList) > minNumberOfMatches:
                imageClassifiedToIndex = matchList.index(max(matchList))
        print(matchList)
        return imageClassifiedToIndex

    def classify(self, image):
        descriptorList = self.getDescriptors(self.images)
        print(len(descriptorList))
        id = self.getId(image, descriptorList)
        if id != -1:
            return "Result: " + self.classNames[id]
        else:
            return 'Unknown'
            # cv2.putText(img2, "Result:" + classNames[id], (30, 30), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 1)


    # img1 = cv2.imread("FeatureBased/Classes/victory.png",0)
    # img2 = cv2.imread("FeatureBased/ToDetect/victory_test.png",0)





    # orb = cv2.ORB_create()
    # keypoints, descriptors = sift.detectAndCompute(img1,None)
    #
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)

    # imgKp1 = cv2.drawKeypoints(img1, kp1, None)
    # imgKp2 = cv2.drawKeypoints(img2, kp2, None)
    # img1 = cv2.drawKeypoints(img1, keypoints, None)
    # cv2.imshow("Image", img1)


    # cv2.imshow('Kp1',imgKp1)
    # cv2.imshow('Kp2',imgKp2)
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)


# cv2.waitKey(0)