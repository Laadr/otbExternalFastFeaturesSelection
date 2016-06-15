cmake_minimum_required(VERSION 2.8.9)
project(OTBFastFeaturesSelection)
set(OTBFastFeaturesSelection_LIBRARIES OTBFastFeaturesSelection)

# # check for OpenMP
# find_package(OpenMP)
# if(OPENMP_FOUND)
#   message(STATUS "Enabling OpenMP support")
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# else()
#   message(STATUS "Disabling OpenMP support")
# endif()

if(NOT OTB_SOURCE_DIR)
  find_package(OTB REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${OTB_CMAKE_DIR})
  include(OTBModuleExternal)
else()
  otb_module_impl()
endif()

