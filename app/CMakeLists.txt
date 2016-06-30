set(OTBFastFeaturesSelection_LINK_LIBS
  ${OTBVectorDataBase_LIBRARIES}
  ${OTBConversion_LIBRARIES}
  ${OTBStatistics_LIBRARIES}
  ${OTBColorMap_LIBRARIES}
  ${OTBBoost_LIBRARIES}
  ${OTBInterpolation_LIBRARIES}
  ${OTBMajorityVoting_LIBRARIES}
  ${OTBVectorDataIO_LIBRARIES}
  ${OTBSOM_LIBRARIES}
  ${OTBSVMLearning_LIBRARIES}
  ${OTBLearningBase_LIBRARIES}
  ${OTBApplicationEngine_LIBRARIES}
  ${OTBIndices_LIBRARIES}
  ${OTBMathParser_LIBRARIES}
  ${OTBDempsterShafer_LIBRARIES}
  ${OTBGdalAdapters_LIBRARIES}
  ${OTBProjection_LIBRARIES}
  ${OTBImageBase_LIBRARIES}
  ${OTBIOXML_LIBRARIES}
  ${OTBVectorDataManipulation_LIBRARIES}
  ${OTBStreaming_LIBRARIES}
  ${OTBImageManipulation_LIBRARIES}
  ${OTBObjectList_LIBRARIES}
  ${OTBCommon_LIBRARIES}
)

OTB_CREATE_APPLICATION(NAME TrainGMMApp
                       SOURCES otbTrainGMMApp.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES}
)

# OTB_CREATE_APPLICATION(NAME TrainGMMSelectionApp
#                        SOURCES otbTrainGMMSelectionApp.cxx
#                        LINK_LIBRARIES ${${otb-module}_LIBRARIES}
# )

# OTB_CREATE_APPLICATION(NAME PredictGMMApp
#                        SOURCES otbPredictGMMApp.cxx
#                        LINK_LIBRARIES ${${otb-module}_LIBRARIES}
# )
