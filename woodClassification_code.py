# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 09:39:23 2021

@author: Alicia

code to 
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import pandas as pd
import time
import arcpy

arcpy.CheckOutExtension("SpatialAnalyst") 
arcpy.CheckOutExtension("ImageAnalyst") 

#####INCLUDES SEGMENTATION####################################################################################################################################
#"""Mackenzie Coast SVM Multi-Class classification"""
#"""train svm classifier and classify raster"""
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Mack_WV02_20170727211656_Coast_nd.tif"
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/mackCoast/mackCoast_samples_Training.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#parameter="20_16_10"
#sampling="EQUALIZED_STRATIFIED_RANDOM"
#
#
##parameter=["16_4_70","16_10_40","16_16_10","18_10_40","18_16_70","18_4_10","20_16_10","20_10_40","20_4_70"]
##sampling = ["STRATIFIED_RANDOM","EQUALIZED_STRATIFIED_RANDOM","RANDOM"]
#
#
#name_seg = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Mack_Coast_Loop_rbg/Mack_Coast_Seg_" + parameter + ".tif"
#inSegRaster = name_seg
#    
#name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/mackCoast/Classifier_SVM_mackCoast_" + parameter + "_ab.ecd"
#definition_file= name_def
#
##Execute
#arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#        maxNumSamples, attributes)
#
#    
#classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/mackCoast/classified_mackCoast_" + parameter + "_ab.tif"
#classifiedraster.save(classified_file)
#    
#del name_seg, inSegRaster, name_def, definition_file, classifiedraster
#
#"""create accuracy assessment points"""
####files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/mackCoast/mackCoast_samples_Reference.shp"
#    
#name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackCoast/mackCoast_Accuracy_" + parameter + "_" + sampling + "_ab.shp"
#points_acc = name_points_acc
#    
#target_field = "GROUND_TRUTH"
#num_random_points = "500"
##sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#del in_ref_data, name_points_acc, target_field, num_random_points
#
#"""update accuracy assessment points """
####files needed: classified data, accuracy points, updates accuracy points, target field
#    
#name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackCoast/mackCoast_Accuracy_Update_" + parameter + "_" + sampling + "_ab.shp"
#points_up_acc = name_points_up_acc
#
#target_field = "CLASSIFIED"
#
#arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#del target_field, name_points_up_acc, points_acc
#
#"""compute confusion matrix"""
####files needed: updated accuracy assessment points, output confusion matrix
#
#name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackCoast/mackCoast_ConfusionMatrix_" + parameter + "_" + sampling + "_ab.dbf"
#confusion_matrix = name_con_matrix
#
#arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del parameter, sampling
#
#del training_samples, inRaster, maxNumSamples, attributes
######################################################################################################################################################################
#"""Mackenzie Lake"""
#"""train svm classifier and classify raster"""
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Mack_GEO1_20170624204908_Inuvik_nd.tif"
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/mackLake/mackLake_samples_Training.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#parameter="20_16_10"
#sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#
##parameter=["16_4_70","16_10_40","16_16_10","18_10_40","18_16_70","18_4_10","20_16_10","20_10_40","20_4_70"]
##sampling = ["STRATIFIED_RANDOM","EQUALIZED_STRATIFIED_RANDOM","RANDOM"]
#
#
#name_seg = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Mack_Lake_Loop_rbg/Mack_Lake_Seg_" + parameter + ".tif"
#inSegRaster = name_seg
#    
#name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/mackLake/Classifier_SVM_mackLake_" + parameter + "_ab.ecd"
#definition_file= name_def
#
##Execute
#arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#        maxNumSamples, attributes)
#
#    
#classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/mackLake/classified_mackLake_" + parameter + "_ab.tif"
#classifiedraster.save(classified_file)
#    
#del name_seg, inSegRaster, name_def, definition_file, classifiedraster
#
#"""create accuracy assessment points"""
####files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/mackLake/mackLake_samples_Reference.shp"
#    
#name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackLake/mackLake_Accuracy_" + parameter + "_" + sampling + "_ab.shp"
#points_acc = name_points_acc
#    
#target_field = "GROUND_TRUTH"
#num_random_points = "500"
##sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#del in_ref_data, name_points_acc, target_field, num_random_points
#
#"""update accuracy assessment points """
####files needed: classified data, accuracy points, updates accuracy points, target field
#    
#name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackLake/mackLake_Accuracy_Update_" + parameter + "_" + sampling + "_ab.shp"
#points_up_acc = name_points_up_acc
#
#target_field = "CLASSIFIED"
#
#arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#del target_field, name_points_up_acc, points_acc
#
#"""compute confusion matrix"""
####files needed: updated accuracy assessment points, output confusion matrix
#
#name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackLake/mackLake_ConfusionMatrix_" + parameter + "_" + sampling + "_ab.dbf"
#confusion_matrix = name_con_matrix
#
#arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del parameter, sampling
#
#del training_samples, inRaster, maxNumSamples, attributes
######################################################################################################################################################################
#"""Slave Raft"""
#"""train svm classifier and classify raster"""
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Slave_WV02_20170720185847_Raft_nd.tif"
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/slaveRaft/slaveRaft_samples_Training.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#parameter="20_16_10"
#sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#
##parameter=["16_4_70","16_10_40","16_16_10","18_10_40","18_16_70","18_4_10","20_16_10","20_10_40","20_4_70"]
##sampling = ["STRATIFIED_RANDOM","EQUALIZED_STRATIFIED_RANDOM","RANDOM"]
#
#
#name_seg = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Slave_Raft_Loop_rbg/Slave_Raft_Seg_" + parameter + ".tif"
#inSegRaster = name_seg
#    
#name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/slaveRaft/Classifier_SVM_slaveRaft_" + parameter + "_ab.ecd"
#definition_file= name_def
#
##Execute
#arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#        maxNumSamples, attributes)
#
#    
#classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/slaveRaft/classified_slaveRaft_" + parameter + "_ab.tif"
#classifiedraster.save(classified_file)
#    
#del name_seg, inSegRaster, name_def, definition_file, classifiedraster
#
#"""create accuracy assessment points"""
####files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/slaveRaft/slaveRaft_samples_Reference.shp"
#    
#name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/slaveRaft/slaveRaft_Accuracy_" + parameter + "_" + sampling + "_ab.shp"
#points_acc = name_points_acc
#    
#target_field = "GROUND_TRUTH"
#num_random_points = "500"
##sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#del in_ref_data, name_points_acc, target_field, num_random_points
#
#"""update accuracy assessment points """
####files needed: classified data, accuracy points, updates accuracy points, target field
#    
#name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/slaveRaft/slaveRaft_Accuracy_Update_" + parameter + "_" + sampling + "_ab.shp"
#points_up_acc = name_points_up_acc
#
#target_field = "CLASSIFIED"
#
#arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#del target_field, name_points_up_acc, points_acc
#
#"""compute confusion matrix"""
####files needed: updated accuracy assessment points, output confusion matrix
#
#name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/slaveRaft/slaveRaft_ConfusionMatrix_" + parameter + "_" + sampling + "_ab.dbf"
#confusion_matrix = name_con_matrix
#
#arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del parameter, sampling
#
#del training_samples, inRaster, maxNumSamples, attributes
#####################################################################################################################################################################



####NO SEGMENTATION####################################################################################################################################
#"""Mackenzie Coast SVM Multi-Class classification"""
#"""train svm classifier and classify raster"""
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Mack_WV02_20170727211656_Coast_nd.tif"
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/mackCoast/mackCoast_samples_Training.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#sampling="EQUALIZED_STRATIFIED_RANDOM"
#
#name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/mackCoast/Classifier_SVM_mackCoast_" + "ab.ecd"
#definition_file= name_def
#
##Execute
#arcpy.ia.TrainSupportVectorMachineClassifier(inRaster, training_samples, definition_file, "", 
#        maxNumSamples, attributes)
#
#    
#classifiedraster = arcpy.ia.ClassifyRaster(inRaster, definition_file)
#    
#classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/mackCoast/classified_mackCoast_" + "_ab.tif"
#classifiedraster.save(classified_file)
#    
#del name_def, definition_file, classifiedraster
#
#"""create accuracy assessment points"""
####files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/mackCoast/mackCoast_samples_Reference.shp"
#    
#name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackCoast/mackCoast_Accuracy_" + sampling + "_ab.shp"
#points_acc = name_points_acc
#    
#target_field = "GROUND_TRUTH"
#num_random_points = "500"
##sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#del in_ref_data, name_points_acc, target_field, num_random_points
#
#"""update accuracy assessment points """
####files needed: classified data, accuracy points, updates accuracy points, target field
#    
#name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackCoast/mackCoast_Accuracy_Update_" + sampling + "_ab.shp"
#points_up_acc = name_points_up_acc
#
#target_field = "CLASSIFIED"
#
#arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#del target_field, name_points_up_acc, points_acc
#
#"""compute confusion matrix"""
####files needed: updated accuracy assessment points, output confusion matrix
#
#name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackCoast/mackCoast_ConfusionMatrix_" + sampling + "_ab.dbf"
#confusion_matrix = name_con_matrix
#
#arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del sampling
#
#del training_samples, inRaster, maxNumSamples, attributes
#
######################################################################################################################################################################
#"""Mackenzie Lake"""
#"""train svm classifier and classify raster"""
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Mack_GEO1_20170624204908_Inuvik_nd.tif"
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/mackLake/mackLake_samples_Training.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#sampling = "EQUALIZED_STRATIFIED_RANDOM"
#  
#name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/mackLake/Classifier_SVM_mackLake_" + "ab.ecd"
#definition_file= name_def
#
##Execute
#arcpy.ia.TrainSupportVectorMachineClassifier(inRaster, training_samples, definition_file, "", 
#        maxNumSamples, attributes)
#
#    
#classifiedraster = arcpy.ia.ClassifyRaster(inRaster, definition_file)
#    
#classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/mackLake/classified_mackLake_" + "ab.tif"
#classifiedraster.save(classified_file)
#    
#del name_def, definition_file, classifiedraster
#
#"""create accuracy assessment points"""
####files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/mackLake/mackLake_samples_Reference.shp"
#    
#name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackLake/mackLake_Accuracy_" + sampling + "_ab.shp"
#points_acc = name_points_acc
#    
#target_field = "GROUND_TRUTH"
#num_random_points = "500"
##sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#del in_ref_data, name_points_acc, target_field, num_random_points
#
#"""update accuracy assessment points """
####files needed: classified data, accuracy points, updates accuracy points, target field
#    
#name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackLake/mackLake_Accuracy_Update_" + sampling + "_ab.shp"
#points_up_acc = name_points_up_acc
#
#target_field = "CLASSIFIED"
#
#arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#del target_field, name_points_up_acc, points_acc
#
#"""compute confusion matrix"""
####files needed: updated accuracy assessment points, output confusion matrix
#
#name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/mackLake/mackLake_ConfusionMatrix_" + sampling + "_ab.dbf"
#confusion_matrix = name_con_matrix
#
#arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del sampling
#
#del training_samples, inRaster, maxNumSamples, attributes
######################################################################################################################################################################
#"""Slave Raft"""
#"""train svm classifier and classify raster"""
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Slave_WV02_20170720185847_Raft_nd.tif"
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/slaveRaft/slaveRaft_samples_Training.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#sampling = "EQUALIZED_STRATIFIED_RANDOM"
#   
#name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/slaveRaft/Classifier_SVM_slaveRaft_" + "ab.ecd"
#definition_file= name_def
#
##Execute
#arcpy.ia.TrainSupportVectorMachineClassifier(inRaster, training_samples, definition_file, "", 
#        maxNumSamples, attributes)
#
#    
#classifiedraster = arcpy.ia.ClassifyRaster(inRaster, definition_file)
#    
#classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/slaveRaft/classified_slaveRaft_" + "ab.tif"
#classifiedraster.save(classified_file)
#    
#del name_def, definition_file, classifiedraster
#
#"""create accuracy assessment points"""
####files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/slaveRaft/slaveRaft_samples_Reference.shp"
#    
#name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/slaveRaft/slaveRaft_Accuracy_" + sampling + "_ab.shp"
#points_acc = name_points_acc
#    
#target_field = "GROUND_TRUTH"
#num_random_points = "500"
##sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#del in_ref_data, name_points_acc, target_field, num_random_points
#
#"""update accuracy assessment points """
####files needed: classified data, accuracy points, updates accuracy points, target field
#    
#name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/slaveRaft/slaveRaft_Accuracy_Update_" + sampling + "_ab.shp"
#points_up_acc = name_points_up_acc
#
#target_field = "CLASSIFIED"
#
#arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#del target_field, name_points_up_acc, points_acc
#
#"""compute confusion matrix"""
####files needed: updated accuracy assessment points, output confusion matrix
#
#name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/slaveRaft/slaveRaft_ConfusionMatrix_" + sampling + "_ab.dbf"
#confusion_matrix = name_con_matrix
#
#arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del sampling
#
#del training_samples, inRaster, maxNumSamples, attributes
#####################################################################################################################################################################
#for tree in parameter:
#    
#    for blob in sampling: 
#        
#       name_seg = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Mack_Coast_Loop_rbg/Mack_Coast_Seg_" + tree + ".tif"
#       inSegRaster = name_seg
#    
#       name_def = "E:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Mack_Coast/Classifier_SVM_Mack_Coast_" + tree + "_ab.ecd"
#       definition_file= name_def
#
#       #Execute
#       arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#                             maxNumSamples, attributes)
#
#    
#       classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#       classified_file = "E:/RemoteSensing_Analysis/Project_WoodRS/Classified/Mack_Lake_Inuvik/Classified_Mack_Lake_" + tree + "_ab.tif"
#       classifiedraster.save(classified_file)
#    
#       del name_seg, inSegRaster, name_def, definition_file, classifiedraster
#    
#        """create accuracy assessment points"""
#        ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#        in_ref_data = "E:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Mack_Lake_Inuvik/ReferenceSamples_Inuvik.shp"
#    
#        name_points_acc = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_Inuvik/Mack_Lake_Accuracy_" + tree + "_" + blob + "_ab.shp"
#        points_acc = name_points_acc
#    
#        target_field = "GROUND_TRUTH"
#        num_random_points = "500"
#        #sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#        arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,blob)
#
#        del in_ref_data, name_points_acc, target_field, num_random_points
#
#        """update accuracy assessment points """
#        ###files needed: classified data, accuracy points, updates accuracy points, target field
#    
#        name_points_up_acc = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_Inuvik/Mack_Lake_Accuracy_Update_" + tree + "_" + blob + "_ab.shp"
#        points_up_acc = name_points_up_acc
#
#        target_field = "CLASSIFIED"
#
#        arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#        del target_field, name_points_up_acc, points_acc
#
#        """compute confusion matrix"""
#        ###files needed: updated accuracy assessment points, output confusion matrix
#
#        name_con_matrix = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_Inuvik/Mack_Lake_ConfusionMatrix_" + tree + "_" + blob + "_ab.dbf"
#        confusion_matrix = name_con_matrix
#
#        arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#        del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del tree, parameter, blob, sampling
#
#del tree, training_samples, inRaster, maxNumSamples, attributes, parameter, sampling
#

######################################
######################################
#####################################

#### for Slave Raft ####
#""" segment images- create 9 segmented images"""
#
#inRaster = "E:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Slave_WV02_20170720185847_Raft_nd.tif"
##band_indexes = "5 3 2"
##
##parameters = [["16","4","70"],["16","10","40"],["16","16","10"],["18","10","40"],["18","16","70"],["18","4","10"],["20","16","10"],["20","10","40"],["20","4","70"]]
##
##
##for tree in parameters:
##    
##            spectral_detail = tree[0]
##            spatial_detail = tree[1]
##            min_segment_size = tree[2]
##    
##            # Execute 
##            seg_raster = arcpy.ia.SegmentMeanShift(inRaster, spectral_detail, spatial_detail, min_segment_size, band_indexes)
##
##            name = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Slave_Raft_Loop_rbg/Slave_Raft_Seg_" + spectral_detail + "_" + spatial_detail + "_" + min_segment_size + ".tif"
##            #Save the output 
##            seg_raster.save(name)
##
##del parameters, tree, spectral_detail, spatial_detail, min_segment_size, seg_raster, band_indexes, name 
#
#"""train svm classifier and classify raster"""
#
#training_samples = "E:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/SLave_Raft/TrainingSamples_Raft.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#
#parameter=["16_4_70","16_10_40","16_16_10","18_10_40","18_16_70","18_4_10","20_16_10","20_10_40","20_4_70"]
#
#for tree in parameter:
#    
#    name_seg = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Slave_Raft_Loop_rbg/Slave_Raft_Seg_" + tree + ".tif"
#    inSegRaster = name_seg
#    
#    name_def = "E:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Slave_Raft/Classifier_SVM_Slave_Raft_" + tree + "_ab.ecd"
#    definition_file= name_def
#
#    #Execute
#    arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#                             maxNumSamples, attributes)
#
#    
#    classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#    classified_file = "E:/RemoteSensing_Analysis/Project_WoodRS/Classified/Slave_Raft/Classified_Slave_Raft_" + tree + "_ab.tif"
#    classifiedraster.save(classified_file)
#    
#    del name_seg, inSegRaster, name_def, definition_file, classifiedraster
#    
#    """create accuracy assessment points"""
#        ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#    in_ref_data = "E:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Slave_Raft/ReferenceSamples_Raft.shp"
#    
#    name_points_acc = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Slave_Raft/Slave_Raft_Accuracy_" + tree + "_ab.shp"
#    points_acc = name_points_acc
#    
#    target_field = "GROUND_TRUTH"
#    num_random_points = "500"
#    sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#    arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#    del in_ref_data, name_points_acc, target_field, num_random_points, sampling
#
#    """update accuracy assessment points """
#        ###files needed: classified data, accuracy points, updates accuracy points, target field
#    
#    name_points_up_acc = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Slave_Raft/Slave_Raft_Accuracy_Update_" + tree + "_ab.shp"
#    points_up_acc = name_points_up_acc
#
#    target_field = "CLASSIFIED"
#
#    arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#    del target_field, name_points_up_acc, points_acc
#
#    """compute confusion matrix"""
#        ###files needed: updated accuracy assessment points, output confusion matrix
#
#    name_con_matrix = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Slave_Raft/Slave_Raft_ConfusionMatrix_" + tree + "_ab.dbf"
#    confusion_matrix = name_con_matrix
#
#    arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#    del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del tree, training_samples, inRaster, maxNumSamples, attributes, parameter
#

######################################
######################################
#####################################
#### for Alaska Bar2 ####
""" segment images- create 9 segmented images"""
#
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Alaska2_GEO1_20180903204046_Bar_nd.tif"
#band_indexes = "3 2 1"
#
##parameters = [["16","4","70"],["16","10","40"],["16","16","10"],["18","10","40"],["18","16","70"],["18","4","10"],["20","16","10"],["20","10","40"],["20","4","70"]]
#parameters = [["20","16","5"]]
#
#
#for tree in parameters:
#    
#            spectral_detail = tree[0]
#            spatial_detail = tree[1]
#            min_segment_size = tree[2]
#    
#            # Execute 
#            seg_raster = arcpy.ia.SegmentMeanShift(inRaster, spectral_detail, spatial_detail, min_segment_size, band_indexes)
#
#            name = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Alaska_Bar2_Loop_bc/Alaska_Bar2_Seg_" + spectral_detail + "_" + spatial_detail + "_" + min_segment_size + "_321.tif"
#            #Save the output 
#            seg_raster.save(name)
#
#del parameters, tree, spectral_detail, spatial_detail, min_segment_size, seg_raster, band_indexes, name 
#
#"""train svm classifier and classify raster"""
#
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Alaska_Bar2/TrainingSamples_Bar2.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#
#parameter=["20_16_5"]
#
#for tree in parameter:
#    
#    name_seg = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Alaska_Bar2_Loop_bc/Alaska_Bar2_Seg_" + tree + "_321.tif"
#    inSegRaster = name_seg
#    
#    name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Alaska_Bar2_bc/Classifier_SVM_Alaska_Bar2_" + tree + "_321_ab.ecd"
#    definition_file= name_def
#
#    #Execute
#    arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#                             maxNumSamples, attributes)
#
#    
#    classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#    classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Classified/Alaska_Bar2_bc/Classified_Alaska_Bar2_" + tree + "_321_ab.tif"
#    classifiedraster.save(classified_file)
#    
#    del name_seg, inSegRaster, name_def, definition_file, classifiedraster
#    
#    """create accuracy assessment points"""
#        ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#    in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Alaska_Bar2/ReferenceSamples_Bar2.shp"
#    
#    name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar2_bc/Alaska_Bar2_Accuracy_" + tree + "_321_ab.shp"
#    points_acc = name_points_acc
#    
#    target_field = "GROUND_TRUTH"
#    num_random_points = "500"
#    sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#    arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#    del in_ref_data, name_points_acc, target_field, num_random_points, sampling
#
#    """update accuracy assessment points """
#        ###files needed: classified data, accuracy points, updates accuracy points, target field
#    
#    name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar2_bc/Alaska_Bar2_Accuracy_Update_" + tree + "_321_ab.shp"
#    points_up_acc = name_points_up_acc
#
#    target_field = "CLASSIFIED"
#
#    arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#    del target_field, name_points_up_acc, points_acc
#
#    """compute confusion matrix"""
#        ###files needed: updated accuracy assessment points, output confusion matrix
#
#    name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar2_bc/Alaska_Bar2_ConfusionMatrix_" + tree + "_321_ab.dbf"
#    confusion_matrix = name_con_matrix
#
#    arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#    del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del tree, training_samples, inRaster, maxNumSamples, attributes, parameter
#
#####################################
#####################################
#
##### for Alaska Bar1 ####
#""" segment images- create 9 segmented images"""
#
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Alaska1_WV03_20180519205325_Bar_nd.tif"
#band_indexes = "5 3 2"
#
##parameters = [["16","4","70"],["16","10","40"],["16","16","10"],["18","10","40"],["18","16","70"],["18","4","10"],["20","16","10"],["20","10","40"],["20","4","70"]]
#parameters = [["20","16","5"]]
#
#
#for tree in parameters:
#    
#            spectral_detail = tree[0]
#            spatial_detail = tree[1]
#            min_segment_size = tree[2]
#    
#            # Execute 
#            seg_raster = arcpy.ia.SegmentMeanShift(inRaster, spectral_detail, spatial_detail, min_segment_size, band_indexes)
#
#            name = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Alaska_Bar1_Loop_bc/Alaska_Bar1_Seg_" + spectral_detail + "_" + spatial_detail + "_" + min_segment_size + "_532.tif"
#            #Save the output 
#            seg_raster.save(name)
#
#del parameters, tree, spectral_detail, spatial_detail, min_segment_size, seg_raster, band_indexes, name 
#
#"""train svm classifier and classify raster"""
#
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Alaska_Bar1/TrainingSamples_Alaska1.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#
#parameter=["20_16_5"]
#
#for tree in parameter:
#    
#    name_seg = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Alaska_Bar1_Loop_bc/Alaska_Bar1_Seg_" + tree + "_532.tif"
#    inSegRaster = name_seg
#    
#    name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Alaska_Bar1_bc/Classifier_SVM_Alaska_Bar1_" + tree + "_532_ab.ecd"
#    definition_file= name_def
#
#    #Execute
#    arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#                             maxNumSamples, attributes)
#
#    
#    classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#    classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Classified/Alaska_Bar1_bc/Classified_Alaska_Bar1_" + tree + "_532_ab.tif"
#    classifiedraster.save(classified_file)
#    
#    del name_seg, inSegRaster, name_def, definition_file, classifiedraster
#    
#    """create accuracy assessment points"""
#        ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#    in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Alaska_Bar1/ReferenceSamples_Alaska1.shp"
#    
#    name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar1_bc/Alaska_Bar1_Accuracy_" + tree + "_532_ab.shp"
#    points_acc = name_points_acc
#    
#    target_field = "GROUND_TRUTH"
#    num_random_points = "500"
#    sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#    arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#    del in_ref_data, name_points_acc, target_field, num_random_points, sampling
#
#    """update accuracy assessment points """
#        ###files needed: classified data, accuracy points, updates accuracy points, target field
#    
#    name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar1_bc/Alaska_Bar1_Accuracy_Update_" + tree + "_532_ab.shp"
#    points_up_acc = name_points_up_acc
#
#    target_field = "CLASSIFIED"
#
#    arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#    del target_field, name_points_up_acc, points_acc
#
#    """compute confusion matrix"""
#        ###files needed: updated accuracy assessment points, output confusion matrix
#
#    name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar1_bc/Alaska_Bar1_ConfusionMatrix_" + tree + "_532_ab.dbf"
#    confusion_matrix = name_con_matrix
#
#    arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#    del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del tree, training_samples, inRaster, maxNumSamples, attributes, parameter

####################################
####################################

#"""use other parameters for svm classifier"""
#### for inuvik site
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Mack_GEO1_20170624204908_Inuvik_nd.tif"
#
#"""train svm classifier and classify raster"""
#
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Mack_Lake_Inuvik/TrainingSamples_Inuvik.shp"
#maxNumSamples = "0"
##attributes = ["COLOR;MEAN","COLOR;MEAN;STD","COLOR;MEAN;STD;COUNT","COLOR;MEAN;STD;COUNT;COMPACTNESS","COLOR;MEAN;STD;COUNT;COMPACTNESS;RECTANGULARITY"]
#attributes = ["COLOR","MEAN","STD","COUNT","COMPACTNESS","RECTANGULARITY"]
#
##n=0
#
##parameter=["16_4_70","18_10_40","20_16_10"]
#parameter=["20_16_10"]
#
#for tree in parameter:
#    
#    name_seg = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Mack_Lake_Loop_rbg/Mack_Lake_Seg_" + tree + ".tif"
#    inSegRaster = name_seg
#    n=0     
#    
#    for blob in attributes:
#
#        n=n+1    
#    
#        name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Mack_Lake_SVM_Loop2/Classifier_SVM_Mack_Lake_" + tree + "_" + str(n) + ".ecd"
#        definition_file= name_def
#
#        #Execute
#        arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#                             maxNumSamples, blob)
#
#    
#        classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#        classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Classified/Mack_Lake_SVM_Loop2/Classified_Mack_Lake_" + tree + "_" + str(n) + ".tif"
#        classifiedraster.save(classified_file)
#    
#        del name_def, definition_file, classifiedraster
#    
#        """create accuracy assessment points"""
#        ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#        in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Mack_Lake_Inuvik/ReferenceSamples_Inuvik.shp"
#    
#        name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_SVM_Loop2/Mack_Lake_Accuracy_" + tree + "_" + str(n) + ".shp"
#        points_acc = name_points_acc
#    
#        target_field = "GROUND_TRUTH"
#        num_random_points = "500"
#        sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#        arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#        del in_ref_data, name_points_acc, target_field, num_random_points, sampling
#        
#        """update accuracy assessment points """
#        ###files needed: classified data, accuracy points, updates accuracy points, target field
#    
#        name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_SVM_Loop2/Mack_Lake_Accuracy_Update_" + tree + "_" + str(n) + ".shp"
#        points_up_acc = name_points_up_acc
#
#        target_field = "CLASSIFIED"
#
#        arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#        del target_field, name_points_up_acc, points_acc
#
#        """compute confusion matrix"""
#        ###files needed: updated accuracy assessment points, output confusion matrix
#
#        name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_SVM_Loop2/Mack_Lake_ConfusionMatrix_" + tree + "_" + str(n) + ".dbf"
#        confusion_matrix = name_con_matrix
#
#        arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#        del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#    del name_seg, inSegRaster, n
#    
#del tree, training_samples, inRaster, maxNumSamples, attributes, parameter, blob

#####################################
#####################################3

#"""use other parameters for svm classifier"""
#### for Alaska Bar1
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Alaska1_WV03_20180519205325_Bar_nd.tif"
#
#"""train svm classifier and classify raster"""
#
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Alaska_Bar1/TrainingSamples_Alaska1.shp"
#maxNumSamples = "0"
##attributes = ["COLOR;MEAN","COLOR;MEAN;STD","COLOR;MEAN;STD;COUNT","COLOR;MEAN;STD;COUNT;COMPACTNESS","COLOR;MEAN;STD;COUNT;COMPACTNESS;RECTANGULARITY"]
##attributes = ["COLOR","MEAN","STD","COUNT","COMPACTNESS","RECTANGULARITY"]
#attributes = ["MEAN","COLOR;MEAN","MEAN;STD","MEAN;COUNT","MEAN;COMPACTNESS","MEAN;RECTANGULARITY"]
#
##parameter=["16_4_70","18_10_40","20_16_10"]
#parameter=["20_16_10"]
#
#for tree in parameter:
#    
#    name_seg = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Alaska_Bar1_Loop_rbg/Alaska_Bar1_Seg_" + tree + ".tif"
#    inSegRaster = name_seg
#    n=0
#    
#    for blob in attributes:
#        
#        n=n+1    
#    
#        name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Alaska_Bar1_SVM_Loop3/Classifier_SVM_Alaska_Bar1_" + tree + "_"+ str(n) + ".ecd"
#        definition_file= name_def
#
#        #Execute
#        arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#                             maxNumSamples, blob)
#
#    
#        classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#        classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Classified/Alaska_Bar1_SVM_Loop3/Classified_Alaska_Bar1_" + tree + "_" + str(n) + ".tif"
#        classifiedraster.save(classified_file)
#    
#        del name_def, definition_file, classifiedraster
#    
#        """create accuracy assessment points"""
#        ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#        in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Alaska_Bar1/ReferenceSamples_Alaska1.shp"
#    
#        name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar1_SVM_Loop3/Alaska_Bar1_Accuracy_" + tree + "_" + str(n) + ".shp"
#        points_acc = name_points_acc
#    
#        target_field = "GROUND_TRUTH"
#        num_random_points = "500"
#        sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#        arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#        del in_ref_data, name_points_acc, target_field, num_random_points, sampling
#        
#        """update accuracy assessment points """
#        ###files needed: classified data, accuracy points, updates accuracy points, target field
#    
#        name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar1_SVM_Loop3/Alaska_Bar1_Accuracy_Update_" + tree + "_" + str(n) + ".shp"
#        points_up_acc = name_points_up_acc
#
#        target_field = "CLASSIFIED"
#
#        arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#        del target_field, name_points_up_acc, points_acc
#
#        """compute confusion matrix"""
#        ###files needed: updated accuracy assessment points, output confusion matrix
#
#        name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar1_SVM_Loop3/Alaska_Bar1_ConfusionMatrix_" + tree + "_" + str(n) + ".dbf"
#        confusion_matrix = name_con_matrix
#
#        arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#        del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#    del name_seg, inSegRaster, n
#    
#del tree, training_samples, inRaster, maxNumSamples, attributes, parameter, blob

#####################################
#####################################3

#"""use other parameters for svm classifier"""
#### for Slave Raft
#inRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Slave_WV02_20170720185847_Raft_nd.tif"
#
#"""train svm classifier and classify raster"""
#
#training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Slave_Raft/TrainingSamples_Raft.shp"
#maxNumSamples = "0"
##attributes = ["COLOR;MEAN","COLOR;MEAN;STD","COLOR;MEAN;STD;COUNT","COLOR;MEAN;STD;COUNT;COMPACTNESS","COLOR;MEAN;STD;COUNT;COMPACTNESS;RECTANGULARITY"]
##attributes = ["COLOR","MEAN","STD","COUNT","COMPACTNESS","RECTANGULARITY"]
#attributes = ["MEAN","COLOR;MEAN","MEAN;STD","MEAN;COUNT","MEAN;COMPACTNESS","MEAN;RECTANGULARITY"]
#
##parameter=["16_4_70","18_10_40","20_16_10"]
#parameter=["20_16_10"]
#
#for tree in parameter:
#    
#    name_seg = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Slave_Raft_Loop_rbg/Slave_Raft_Seg_" + tree + ".tif"
#    inSegRaster = name_seg
#    n=0 
#    
#    for blob in attributes:
#        
#        n=n+1    
#    
#        name_def = "D:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Slave_Raft_SVM_Loop3/Classifier_SVM_Slave_Raft_" + tree + "_" + str(n) + ".ecd"
#        definition_file= name_def
#
#        #Execute
#        arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#                             maxNumSamples, blob)
#
#    
#        classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#        classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Classified/Slave_Raft_SVM_Loop3/Classified_Slave_Raft_" + tree + "_" + str(n) + ".tif"
#        classifiedraster.save(classified_file)
#    
#        del name_def, definition_file, classifiedraster
#    
#        """create accuracy assessment points"""
#        ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#        in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Slave_Raft/ReferenceSamples_Raft.shp"
#    
#        name_points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Slave_Raft_SVM_Loop3/Slave_Raft_Accuracy_" + tree + "_" + str(n) + ".shp"
#        points_acc = name_points_acc
#    
#        target_field = "GROUND_TRUTH"
#        num_random_points = "500"
#        sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#        arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#        del in_ref_data, name_points_acc, target_field, num_random_points, sampling
#        
#        """update accuracy assessment points """
#        ###files needed: classified data, accuracy points, updates accuracy points, target field
#    
#        name_points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Slave_Raft_SVM_Loop3/Slave_Raft_Accuracy_Update_" + tree + "_" + str(n) + ".shp"
#        points_up_acc = name_points_up_acc
#
#        target_field = "CLASSIFIED"
#
#        arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#        del target_field, name_points_up_acc, points_acc
#
#        """compute confusion matrix"""
#        ###files needed: updated accuracy assessment points, output confusion matrix
#
#        name_con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Slave_Raft_SVM_Loop3/Slave_Raft_ConfusionMatrix_" + tree + "_" + str(n) + ".dbf"
#        confusion_matrix = name_con_matrix
#
#        arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#        del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#    del name_seg, inSegRaster, n
#    
#del tree, training_samples, inRaster, maxNumSamples, attributes, parameter, blob

############################
##############################
##############################

### for Alaska Bar1 -- testing different band combinations####
#""" segment images- create 9 segmented images"""
#
#inRaster = "E:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Alaska1_WV03_20180519205325_Bar_nd.tif"
#
#training_samples = "E:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Alaska_Bar1/TrainingSamples_Alaska1.shp"
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#
##band_indexes = ["8 3 2","4 1 3","8 4 1"]
#band_indexes = ["4 1 3","8 4 1"]
#parameters = [["16","4","70"],["18","10","40"],["20","16","10"]]
#
#for blob in band_indexes: 
#    for tree in parameters:
#    
#            spectral_detail = tree[0]
#            spatial_detail = tree[1]
#            min_segment_size = tree[2]
#    
#            # Execute 
#            seg_raster = arcpy.ia.SegmentMeanShift(inRaster, spectral_detail, spatial_detail, min_segment_size, blob)
#
#            name = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Alaska_Bar1_Loop_bc/Alaska_Bar1_Seg_" + spectral_detail + "_" + spatial_detail + "_" + min_segment_size + "_" + blob + ".tif"
#            #Save the output 
#            seg_raster.save(name)
#
#    del tree, spectral_detail, spatial_detail, min_segment_size, seg_raster, name 
#
#    """train svm classifier and classify raster"""
#
#    parameter=["16_4_70","18_10_40","20_16_10"]
#
#    for tree in parameter:
#    
#        name_seg = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Alaska_Bar1_Loop_bc/Alaska_Bar1_Seg_" + tree + "_" + blob + ".tif"
#        inSegRaster = name_seg
#    
#        name_def = "E:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Alaska_Bar1_bc/Classifier_SVM_Alaska_Bar1_" + tree + "_" + blob + "_ab.ecd"
#        definition_file= name_def
#
#        #Execute
#        arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, definition_file, inRaster, 
#                             maxNumSamples, attributes)
#
#    
#        classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, definition_file, inRaster)
#    
#        classified_file = "E:/RemoteSensing_Analysis/Project_WoodRS/Classified/Alaska_Bar1_bc/Classified_Alaska_Bar1_" + tree + "_" + blob + "_ab.tif"
#        classifiedraster.save(classified_file)
#    
#        del name_seg, inSegRaster, name_def, definition_file, classifiedraster
#    
#        """create accuracy assessment points"""
#        ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
#    
#        in_ref_data = "E:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Alaska_Bar1/ReferenceSamples_Alaska1.shp"
#    
#        name_points_acc = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar1_bc/Alaska_Bar1_Accuracy_" + tree + "_" + blob + "_ab.shp"
#        points_acc = name_points_acc
#    
#        target_field = "GROUND_TRUTH"
#        num_random_points = "500"
#        sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#        arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
#
#        del in_ref_data, name_points_acc, target_field, num_random_points, sampling
#
#        """update accuracy assessment points """
#        ###files needed: classified data, accuracy points, updates accuracy points, target field
#    
#        name_points_up_acc = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar1_bc/Alaska_Bar1_Accuracy_Update_" + tree + "_" + blob + "_ab.shp"
#        points_up_acc = name_points_up_acc
#
#        target_field = "CLASSIFIED"
#
#        arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
#
#        del target_field, name_points_up_acc, points_acc
#
#        """compute confusion matrix"""
#        ###files needed: updated accuracy assessment points, output confusion matrix
#
#        name_con_matrix = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Alaska_Bar1_bc/Alaska_Bar1_ConfusionMatrix_" + tree + "_" + blob + "_ab.dbf"
#        confusion_matrix = name_con_matrix
#
#        arcpy.ia.ComputeConfusionMatrix(points_up_acc,confusion_matrix)
#
#        del points_up_acc, name_con_matrix, confusion_matrix, classified_file
#    
#del tree, training_samples, inRaster, maxNumSamples, attributes, parameter, band_indexes, blob
#





#""" segment other images- create 9 segmented images"""
#
####Slave River Raft
#inRaster = "E:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Slave_WV02_20170720185847_Raft_nd.tif"
#band_indexes = "5 3 2"
#
#parameters = [["16","4","70"],["16","10","40"],["16","16","10"],["18","10","40"],["18","16","70"],["18","4","10"],["20","16","10"],["20","10","40"],["20","4","70"]]
#
#
#for tree in parameters:
#    
#            spectral_detail = tree[0]
#            spatial_detail = tree[1]
#            min_segment_size = tree[2]
#    
#            # Execute 
#            seg_raster = arcpy.ia.SegmentMeanShift(inRaster, spectral_detail, spatial_detail, min_segment_size, band_indexes)
#
#            name = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Slave_Raft_Loop_rbg/Slave_Raft_Seg_" + spectral_detail + "_" + spatial_detail + "_" + min_segment_size + ".tif"
#            #Save the output 
#            seg_raster.save(name)
#
#del parameters, tree, spectral_detail, spatial_detail, min_segment_size, seg_raster, inRaster, band_indexes, name 
#
####Alaska Bar 2
#inRaster = "E:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Alaska2_GEO1_20180903204046_Bar_nd.tif"
#band_indexes = "3 2 1"
#
#parameters = [["16","4","70"],["16","10","40"],["16","16","10"],["18","10","40"],["18","16","70"],["18","4","10"],["20","16","10"],["20","10","40"],["20","4","70"]]
#
#
#for tree in parameters:
#    
#            spectral_detail = tree[0]
#            spatial_detail = tree[1]
#            min_segment_size = tree[2]
#    
#            # Execute 
#            seg_raster = arcpy.ia.SegmentMeanShift(inRaster, spectral_detail, spatial_detail, min_segment_size, band_indexes)
#
#            name = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Alaska_Bar2_Loop_rbg/Alaska_Bar2_Seg_" + spectral_detail + "_" + spatial_detail + "_" + min_segment_size + ".tif"
#            #Save the output 
#            seg_raster.save(name)
#
#del parameters, tree, spectral_detail, spatial_detail, min_segment_size, seg_raster, inRaster, band_indexes, name 




#parameter=["16_10_40","16_16_10","18_10_40","18_16_70","18_4_10","20_16_10","20_10_40","20_4_70"]
#
#for tree in parameter:
#    
#    name = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Mack_Lake_Loop_rbg/Mack_Lake_Seg_" + tree + ".tif"
#    inSegRaster = name
#    
#    name2 = "E:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Mack_Lake_Inuvik/Classifier_SVM_Mack_Lake_" + tree + "_ab.ecd"
#    in_def_file= name2
#    
#    classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, in_def_file, additional_raster)
#    
#    name3 = "E:/RemoteSensing_Analysis/Project_WoodRS/Classified/Mack_Lake_Inuvik/Classified_Mack_Lake_" + tree + "_ab.tif"
#    classifiedraster.save(name3)
# 
#del tree, additional_raster, parameter, name, inSegRaster, name2, in_def_file, classifiedraster, name3 
#del additional_raster, name, inSegRaster, name2, in_def_file, classifiedraster, name3 
#
#""" compute accuracy assessment"""
#
#in_class_data = "E:/RemoteSensing_Analysis/Project_WoodRS/TrainingSamples/Mack_Lake_Inuvik/ReferenceSamples_Inuvik.shp"
#out_points = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_Inuvik/Mack_Lake_Accuracy_16_4_70_ab.shp"
#target_field = "GROUND_TRUTH"
#num_random_points = "500"
#sampling = "EQUALIZED_STRATIFIED_RANDOM"
#
#arcpy.ia.CreateAccuracyAssessmentPoints(in_class_data,out_points,target_field,num_random_points,sampling)
#
#del in_class_data, out_points, target_field, num_random_points, sampling
#
#in_class_data = "E:/RemoteSensing_Analysis/Project_WoodRS/Classified/Mack_Lake_Inuvik/Classified_Mack_Lake_16_4_70_ab.tif"
# 
#in_points = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_Inuvik/Mack_Lake_Accuracy_16_4_70_ab.shp"
#
#out_points = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_Inuvik/Mack_Lake_Accuracy_Update_16_4_70_ab.shp"
#
#target_field = "CLASSIFIED"
#
#arcpy.ia.UpdateAccuracyAssessmentPoints(in_class_data,in_points,out_points,target_field)
#
#del in_class_data, in_points, out_points, target_field
#
#in_accuracy_assessment_points = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_Inuvik/Mack_Lake_Accuracy_Update_16_4_70_ab.shp"
#
#out_confusion_matrix = "E:/RemoteSensing_Analysis/Project_WoodRS/Accuracy/Mack_Lake_Inuvik/Mack_Lake_ConfusionMatrix_16_4_70_ab.dbf"
#
#arcpy.ia.ComputeConfusionMatrix(in_accuracy_assessment_points,out_confusion_matrix)


#inSegRaster = "E:/RemoteSensing_Analysis/Geo_20090905/SVM_Python/Segmentation/GE01_20090905211232_1_seg.tif"
#train_features = "E:/RemoteSensing_Analysis/Geo_20090905/Training_CNN/TrainingSamples.shp"
#out_definition = "E:/RemoteSensing_Analysis/Geo_20090905/SVM_Python/Training/GE01_20090905211232_1_sig.ecd"
#in_additional_raster = "E:/RemoteSensing_Analysis/Geo_20090905/Training_CNN/Images/GE01_20090905211232_1.tif"
#maxNumSamples = "10"
#attributes = "COLOR;MEAN;STD;COUNT;COMPACTNESS;RECTANGULARITY"

#"""classify raster"""
#
#for tree in parameter:
#    
#    name = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Mack_Lake_Loop_rbg/Mack_Lake_Seg_" + tree + ".tif"
#    inSegRaster = name
#    
#    
#inSegRaster = "E:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/Mack_Lake_Loop_rbg/Mack_Lake_Seg_16_4_70.tif"
#indef_file =  "E:/RemoteSensing_Analysis/Project_WoodRS/Classifier_Files/Mack_Lake_Inuvik/Classifier_SVM_Mack_Lake_16_4_70_ab.ecd"
#in_additional_raster = "E:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Mack_GEO1_20170624204908_Inuvik_nd.tif"
#
#classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, indef_file, in_additional_raster)
#
#classifiedraster.save("E:/RemoteSensing_Analysis/Project_WoodRS/Classified/Mack_Lake_Inuvik/Classified_Mack_Lake_16_4_70_ab.tif")

# Set local variables
#inSegRaster = "E:/RemoteSensing_Analysis/Geo_20090905/SVM_Python/Segmentation/GE01_20090905211232_1_seg.tif"
#indef_file = "E:/RemoteSensing_Analysis/Geo_20090905/SVM_Python/Training/GE01_20090905211232_1_sig.ecd"
#in_additional_raster = in_additional_raster = "E:/RemoteSensing_Analysis/Geo_20090905/Training_CNN/Images/GE01_20090905211232_1.tif"


# Execute 
#classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, indef_file, 
#                                  in_additional_raster)

#save output
#classifiedraster.save("E:/RemoteSensing_Analysis/Geo_20090905/SVM_Python/Classified/GE01_20090905211232_1_classified.tif")

##########################################################################################################
#start=time.time()
#"""02.23.21 automating image classification for all images and techniques"""
#
#
#inRaster = ["D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Mack_WV02_20170727211656_Coast_nd.tif",
#            "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Mack_GEO1_20170624204908_Inuvik_nd.tif",
#            "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Slave_WV02_20170720185847_Raft_nd.tif",
#            "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Alaska1_WV03_20180519205325_Bar_nd.tif",
#            "D:/RemoteSensing_Analysis/Project_WoodRS/Images_NoData/Alaska2_GEO1_20180903204046_Bar_nd.tif"]
#
#siteNames1=["mackCoast","mackLake","slaveRaft","alaskaBar1","alaskaBar2"]
#siteNames2=["Mack_Coast","Mack_Lake","Slave_Raft","Alaska_Bar1","Alaska_Bar2"]
#
#maxNumSamples = "0"
#attributes = "COLOR;MEAN"
#parameter="20_16_10"
#
#for tree in range(4,5):
#    
##    training_samples = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/" + siteNames1[tree] + "/" + siteNames1[tree] + "_samples_Training.shp"
#
#    inSegRaster = "D:/RemoteSensing_Analysis/Project_WoodRS/Segmentation/" + siteNames2[tree] + "_Loop_rbg/" + siteNames2[tree] + "_Seg_" + parameter + ".tif"
#    
#    """OBJECT BASED SUPERVISED"""
#    """train svm classifier"""
##    def_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/" + siteNames1[tree] + "/classifier_" + siteNames1[tree] + "_seg_" + parameter + "_svm_ab.ecd"
##    
##    #Execute
##    arcpy.ia.TrainSupportVectorMachineClassifier(inSegRaster, training_samples, def_file, inRaster[tree], 
##        maxNumSamples, attributes)
##    
##    """classify raster"""
##    classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, def_file, inRaster[tree])
##    
##    classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/" + siteNames1[tree] + "/classified_" + siteNames1[tree] + "_seg_" + parameter + "_svm_ab.tif"
##    classifiedraster.save(classified_file)
##    
##    del def_file, classifiedraster
##
##    """create accuracy assessment points"""
##    ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
##    
##    in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/" + siteNames1[tree] + "/" + siteNames1[tree] + "_samples_Reference.shp"
##    
##    points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/" + siteNames1[tree] + "/AccPoints_" + siteNames1[tree] + "_seg_" + parameter + "_svm_ab_esr.shp"
##    
##    target_field = "GROUND_TRUTH"
##    num_random_points = "500"
##    sampling = "EQUALIZED_STRATIFIED_RANDOM"
##
##    arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
##
##    del in_ref_data, target_field, num_random_points
##
##    """update accuracy assessment points """
##    ###files needed: classified data, accuracy points, updates accuracy points, target field
##    
##    points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/" + siteNames1[tree] + "/UpAccPoints_" + siteNames1[tree] + "_seg_" + parameter + "_svm_ab_esr.shp"
##    
##    target_field = "CLASSIFIED"
##
##    arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
##
##    del target_field, points_acc
##
##    """compute confusion matrix"""
##    ###files needed: updated accuracy assessment points, output confusion matrix
##
##    con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/ConMatrix/matrix_" + siteNames1[tree] + "_seg_" + parameter + "_svm_ab_esr.dbf"
##
##    arcpy.ia.ComputeConfusionMatrix(points_up_acc,con_matrix)
##
##    del points_up_acc, con_matrix, classified_file
#    
#    ##############################################################################################################################################################
##    """PIXEL BASED SUPERVISED"""
##    """train svm classifier"""
##
##    def_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/" + siteNames1[tree] + "/classifier_" + siteNames1[tree] + "_svm_ab.ecd"
##    
##    #Execute
##    arcpy.ia.TrainSupportVectorMachineClassifier(inRaster[tree], training_samples, def_file, "", maxNumSamples, attributes)
##    
##    """classify raster"""
##    classifiedraster = arcpy.ia.ClassifyRaster(inRaster[tree], def_file)
##    
##    classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/" + siteNames1[tree] + "/classified_" + siteNames1[tree] + "_svm_ab.tif"
##    classifiedraster.save(classified_file)
##    
##    del def_file, classifiedraster
##
##    """create accuracy assessment points"""
##    ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
##    
##    in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/" + siteNames1[tree] + "/" + siteNames1[tree] + "_samples_Reference.shp"
##    
##    points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/" + siteNames1[tree] + "/AccPoints_" + siteNames1[tree] + "_svm_ab_esr.shp"
##    
##    target_field = "GROUND_TRUTH"
##    num_random_points = "500"
##    sampling = "EQUALIZED_STRATIFIED_RANDOM"
##
##    arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)
##
##    del in_ref_data, target_field, num_random_points
##
##    """update accuracy assessment points """
##    ###files needed: classified data, accuracy points, updates accuracy points, target field
##    
##    points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/" + siteNames1[tree] + "/UpAccPoints_" + siteNames1[tree] + "_svm_ab_esr.shp"
##    
##    target_field = "CLASSIFIED"
##
##    arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)
##
##    del target_field, points_acc
##
##    """compute confusion matrix"""
##    ###files needed: updated accuracy assessment points, output confusion matrix
##
##    con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy/ConMatrix/matrix_" + siteNames1[tree] + "_svm_ab_esr.dbf"
##
##    arcpy.ia.ComputeConfusionMatrix(points_up_acc,con_matrix)
##
##    del points_up_acc, con_matrix, classified_file
#    
#    #########################################################################################################################################################
#    """OBJECT BASED UNSUPERVISED"""
#    
#    maxNumClasses = ["4","8","16"]
#    maxIteration = "20"
#    minNumSamples = "10"
#    skipFactor = "5"
#    
#    for blob in maxNumClasses:
#        
#        """train iso cluster classifier"""
#        def_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/" + siteNames1[tree] + "/classifier_" + siteNames1[tree] + "_seg_" + parameter + "_iso_"+ blob + "_ab.ecd"
#        
#        arcpy.ia.TrainIsoClusterClassifier(inSegRaster, blob, def_file, inRaster[tree], maxIteration, minNumSamples, skipFactor, attributes)
#        
#        """classify raster"""
#        classifiedraster = arcpy.ia.ClassifyRaster(inSegRaster, def_file, inRaster[tree])
#    
#        classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/" + siteNames1[tree] + "/classified_" + siteNames1[tree] + "_seg_" + parameter + "_iso_"+ blob + "_ab.tif"
#        classifiedraster.save(classified_file)
#    
#        del def_file, classifiedraster
#    
#        """PIXEL BASED UNSUPERVISED"""
#        
#        """train iso cluster classifier"""
#        def_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classifier/" + siteNames1[tree] + "/classifier_" + siteNames1[tree] + "_iso_"+ blob + "_ab.ecd"
#        
#        arcpy.ia.TrainIsoClusterClassifier(inRaster[tree], blob, def_file, "", maxIteration, minNumSamples, skipFactor, attributes)
#        
#        """classify raster"""
#        classifiedraster = arcpy.ia.ClassifyRaster(inRaster[tree], def_file)
#    
#        classified_file = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/" + siteNames1[tree] + "/classified_" + siteNames1[tree] + "_iso_"+ blob + "_ab.tif"
#        classifiedraster.save(classified_file)
#    
#        del def_file, classifiedraster
#    del maxNumClasses, maxIteration, minNumSamples, skipFactor 
#    
#del inRaster, maxNumSamples, attributes, siteNames1, siteNames2, parameter, tree, sampling
#end=time.time()
#
#print(end-start)
###########################################################################################################################
###########################################################################################################################

siteNames1=["mackCoast","mackLake","slaveRaft","alaskaBar1","alaskaBar2"]

file2=["_iso_16_ab_","_iso_seg_20_16_10_16_ab_","_seg_20_16_10_svm_ab_","_svm_ab_"]



for tree in range(0,5):
    
    file=["reclassified_" + siteNames1[tree] + "_iso_16_ab.tif", "reclassified_" + siteNames1[tree] + "_seg_20_16_10_iso_16_ab.tif",
          "classified_" + siteNames1[tree] + "_seg_20_16_10_svm_ab.tif", "classified_" + siteNames1[tree] + "_svm_ab.tif"]
    
    in_ref_data = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/TrainingSamples/" + siteNames1[tree] + "/" + siteNames1[tree] + "_samples_Reference.shp"
    
    for blob in range(0,4):
        
        """create accuracy assessment points"""
        ###files needed: reference data, output file for accuracy points, target field, no. random points, sampling
    
        target_field = "GROUND_TRUTH"
        num_random_points = "100"
        sampling = "EQUALIZED_STRATIFIED_RANDOM"
    
        points_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy2/Points2/AccPoints_" + siteNames1[tree] + file2[blob] + "esr.shp"

        arcpy.ia.CreateAccuracyAssessmentPoints(in_ref_data,points_acc,target_field,num_random_points,sampling)

        del target_field, num_random_points

        """update accuracy assessment points """
        ###files needed: classified data, accuracy points, updates accuracy points, target field
    
        points_up_acc = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy2/Points2/UpAccPoints_" + siteNames1[tree] + file2[blob] + "esr.shp"
    
        target_field = "CLASSIFIED"
        
        classified_file="D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Classified/" + siteNames1[tree] + "/" + file[blob] 
        
        arcpy.ia.UpdateAccuracyAssessmentPoints(classified_file,points_acc,points_up_acc,target_field)

        del target_field, points_acc, classified_file

        """compute confusion matrix"""
        ###files needed: updated accuracy assessment points, output confusion matrix

        con_matrix = "D:/RemoteSensing_Analysis/Project_WoodRS/Revision/Accuracy2/ConMatrix2/matrix_" + siteNames1[tree] + file2[blob] + "esr.dbf"

        arcpy.ia.ComputeConfusionMatrix(points_up_acc,con_matrix)

        del points_up_acc, con_matrix
    del in_ref_data
del siteNames1, file, file2