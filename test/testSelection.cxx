#include <iostream>
#include <string>
#include <ctime>
#include <fstream>
#include "otbListSampleGenerator.h"
#include "otbGMMMachineLearningModel.h"
#include "otbGMMSelectionMachineLearningModel.h"
#include "otbNormalBayesMachineLearningModel.h"


#include "otbVectorImage.h"
#include "otbImageFileReader.h"
#include "otbVectorData.h"
#include "otbVectorDataFileReader.h"

int testSelection(int argc, char * argv[])
{

    int nbSamples          = 600;
    int nbSampleComponents = 20;
    int nbClasses          = 6;
    int selectedVarNb      = 15;

    // Define relevant variables
    std::vector<int> meaningfullVar(6);
    meaningfullVar[0] = 2;
    meaningfullVar[1] = 5;
    meaningfullVar[2] = 8;
    meaningfullVar[3] = 9;
    meaningfullVar[4] = 15;
    meaningfullVar[5] = 16;

    // Input related typedefs
    typedef float InputValueType;
    typedef itk::VariableLengthVector<InputValueType> InputSampleType;
    typedef itk::Statistics::ListSample<InputSampleType> InputListSampleType;

    // Target related typedefs
    typedef int TargetValueType;
    typedef itk::FixedArray<TargetValueType, 1> TargetSampleType;
    typedef itk::Statistics::ListSample<TargetSampleType> TargetListSampleType;

    InputListSampleType::Pointer InputListSample = InputListSampleType::New();
    InputListSample->SetMeasurementVectorSize( nbSampleComponents );
    TargetListSampleType::Pointer TargetListSample = TargetListSampleType::New();

    itk::Statistics::MersenneTwisterRandomVariateGenerator::Pointer randGen;
    randGen = itk::Statistics::MersenneTwisterRandomVariateGenerator::GetInstance();

    // Filling the two input training lists
    for (int i = 0; i < nbSamples; ++i)
    {
        InputSampleType sample;
        TargetValueType label = (i % nbClasses) + 1;

        // Multi-component sample randomly filled from a normal law for each component
        sample.SetSize(nbSampleComponents);
        for (int itComp = 0; itComp < nbSampleComponents; ++itComp)
        {
            if ((itComp==meaningfullVar[0]) && (label==1))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==meaningfullVar[1]) && (label==2))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==meaningfullVar[2]) && (label==3))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==meaningfullVar[3]) && (label==4))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==meaningfullVar[4]) && (label==5))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==meaningfullVar[5]) && (label==6))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else
                sample[itComp] = randGen->GetNormalVariate(100, 1);
        }
        InputListSample->PushBack(sample);
        TargetListSample->PushBack(label);
    }

    // Allocate classifier
    typedef otb::GMMSelectionMachineLearningModel<InputValueType, TargetValueType> GMMType;
    GMMType::Pointer GMMClassifier = GMMType::New();

    // Full model training
    GMMClassifier->SetInputListSample(InputListSample);
    GMMClassifier->SetTargetListSample(TargetListSample);
    GMMClassifier->Train();

    // Perform selection
    GMMClassifier->Selection("forward","jm",selectedVarNb,2);

    // Read and print selection results
    std::vector<int> selectedInd(GMMClassifier->GetSelectedVar());
    std::vector<double> criterionBestValues(GMMClassifier->GetCriterionBestValues());

    std::cout << "Selected variables: ";
    for (std::vector<int>::iterator it = selectedInd.begin(); it != selectedInd.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << std::endl;

    std::cout << "Criterion evolution: ";
    for (std::vector<double>::iterator it = criterionBestValues.begin(); it != criterionBestValues.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << std::endl;

    // Test if the expected variables as been selected
    for (int i = 0; i < 5; ++i)
    {
        if ( selectedInd[i] != meaningfullVar[0] &&
             selectedInd[i] != meaningfullVar[1] &&
             selectedInd[i] != meaningfullVar[2] &&
             selectedInd[i] != meaningfullVar[3] &&
             selectedInd[i] != meaningfullVar[4] &&
             selectedInd[i] != meaningfullVar[5] )
        {
            std::cout << "Wrong variables selected. Possible problem with in selection process." << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Successful selection." << std::endl;
    return EXIT_SUCCESS;
}
