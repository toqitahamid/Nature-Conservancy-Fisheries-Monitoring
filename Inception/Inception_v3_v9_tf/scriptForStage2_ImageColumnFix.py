#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:29:10 2017

@author: kishore
"""

import pandas as pd
import numpy as np 

listofStg1Files = []
def UpdateFileName(imgFileName):
    updatedFileName = ''
    if imgFileName in listofStg1Files:
        updatedFileName = imgFileName       
    else:
        updatedFileName =  'test_stg2/{}'.format(imgFileName)        
    return updatedFileName
#Path Where Stage 1 sample Submission csv is kept
stage1SamplePath = 'result/stg1/submit.csv'
#Path where your stage two Solution csv is kept 
Stage2ModelPath = 'result/stg2/submit.csv'
df = pd.read_csv(stage1SamplePath)
df2 = pd.read_csv(Stage2ModelPath)

listofStg1Files = list(df['image'].values)

df2['image'] = df2['image'].apply(lambda x: UpdateFileName(x))
#Creates a new CSV in the folder where your stage2 solution csv is there. 
df2.to_csv(Stage2ModelPath.rsplit('/',1)[0]+'/UpdatedSubmission.csv',index=False)
