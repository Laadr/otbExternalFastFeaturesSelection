#!/bin/bash

# KNN (OTB)

otbcli_TrainImagesClassifier -io.il image_utm -io.vd field_data_2008_utm_polygon.shp -io.out model_knn.xml -io.confmatout confu_knn.txt -sample.vtr 0.5 -classifier knn > output_knn.txt

# GMM OPENCV (OTB)

otbcli_TrainImagesClassifier -io.il image_utm -io.vd field_data_2008_utm_polygon.shp -io.out model_bayes.xml -io.confmatout confu_bayes.txt -sample.vtr 0.5 -classifier bayes > output_gmm-opcv.txt

# Random Forest (OTB)

otbcli_TrainImagesClassifier -io.il image_utm -io.vd field_data_2008_utm_polygon.shp -io.out model_rf.xml -io.confmatout confu_rf.txt -sample.vtr 0.5 -classifier rf -classifier.rf.var 50 -classifier.rf.nbtrees 200 -classifier.rf.max 40 > output_rf.txt

# LibSVM linear (OTB)

otbcli_TrainImagesClassifier -io.il image_utm -io.vd field_data_2008_utm_polygon.shp -io.out model_svm.xml -io.confmatout confu_svm.txt -sample.vtr 0.5 -classifier libsvm -classifier.libsvm.opt 1 > output_libsvm-linear.txt

# GMM (Lagrange)

otbcli_TrainGMMApp -io.il image_utm -io.vd field_data_2008_utm_polygon.shp -io.out model_gmm-lagr_optim.xml -io.confmatout confu_gmm-lagr_optim.txt -sample.vtr 0.5 -gmm.tau 1 10 100 1000 10000 100000 > output_gmm-lagr_optim.txt
otbcli_TrainGMMApp -io.il image_utm -io.vd field_data_2008_utm_polygon.shp -io.out model_gmm-lagr.xml -io.confmatout confu_gmm-lagr.txt -sample.vtr 0.5 -gmm.tau 10000 > output_gmm-lagr.txt

# GMM + selection (Lagrange)

otbcli_TrainGMMSelectionApp -io.il image_utm -io.vd field_data_2008_utm_polygon.shp -io.out model_gmmSelect_forward-lagr.xml -io.confmatout confu_gmmSelect_forward-lagr.txt -sample.vtr 0.5 -gmm.varnb 30 -gmm.method forward > output_gmmSelect_forward-lagr.txt
otbcli_TrainGMMSelectionApp -io.il image_utm -io.vd field_data_2008_utm_polygon.shp -io.out model_gmmSelect_sffs-lagr.xml -io.confmatout confu_gmmSelect_sffs-lagr.txt -sample.vtr 0.5 -gmm.varnb 30 -gmm.method sffs > output_gmmSelect_sffs-lagr.txt


