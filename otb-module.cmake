set(DOCUMENTATION "OTB external module for Fast Features Selection.")

# OTB_module() defines the module dependencies in FastFeaturesSelection
# FastFeaturesSelection depends on OTBCommon and OTBApplicationEngine
# The testing module in FastFeaturesSelection depends on OTBTestKernel
# and OTBCommandLine

# define the dependencies of the include module and the tests
otb_module(OTBFastFeaturesSelection
  DEPENDS
    OTBCommon
    OTBApplicationEngine
    OTBSupervised
  TEST_DEPENDS
    OTBTestKernel
    OTBCommandLine
  DESCRIPTION
    "${DOCUMENTATION}"
)
