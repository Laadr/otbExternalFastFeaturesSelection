otb_module_test()

#${otb-module} will be the name of this module and will not need to be
#changed when this module is renamed.

set(${otb-module}Tests
  otbFastFeatureSelectionTestDriver.cxx
  testSaveLoad.cxx
  testSelection.cxx
)

add_executable(otbFastFeatureSelectionTestDriver ${${otb-module}Tests})
target_link_libraries(otbFastFeatureSelectionTestDriver ${${otb-module}-Test_LIBRARIES})
otb_module_target_label(otbFastFeatureSelectionTestDriver)

otb_add_test(NAME TvtestSaveLoad
  COMMAND otbFastFeatureSelectionTestDriver testSaveLoad
  )

otb_add_test(NAME TvtestSelection
  COMMAND otbFastFeatureSelectionTestDriver testSelection
  )
