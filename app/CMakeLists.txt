OTB_CREATE_APPLICATION(NAME TrainGMMApp
                       SOURCES otbTrainGMMApp.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES}
)

OTB_CREATE_APPLICATION(NAME TrainGMMSelectionApp
                       SOURCES otbTrainGMMSelectionApp.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES}
)

OTB_CREATE_APPLICATION(NAME PredictGMMApp
                       SOURCES otbPredictGMMApp.cxx
                       LINK_LIBRARIES ${${otb-module}_LIBRARIES}
)
