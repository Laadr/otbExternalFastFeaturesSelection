set(DOCUMENTATION "OTB external module for Fast Features Selection.")

# OTB_module() defines the module dependencies in FFSforGMM
# FFSforGMM depends on OTBCommon and OTBApplicationEngine
# The testing module in FFSforGMM depends on OTBTestKernel
# and OTBCommandLine

# define the dependencies of the include module and the tests
otb_module(OTBFFSforGMM
  DEPENDS
    OTBIOXML
    OTBStatistics
    OTBSupervised
    OTBApplicationEngine
    OTBCommon
  TEST_DEPENDS
    OTBTestKernel
    OTBCommandLine
  DESCRIPTION
    "${DOCUMENTATION}"
)
