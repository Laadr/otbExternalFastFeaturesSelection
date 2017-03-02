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

int testSaveLoad(int itkNotUsed(argc), char * itkNotUsed(argv)[])
{

    int nbSamples          = 600;
    int nbSampleComponents = 20;
    int nbClasses          = 6;
    int selectedVarNb      = 15;

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
            if ((itComp==2) && (label==1))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==5) && (label==2))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==8) && (label==3))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==9) && (label==4))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==15) && (label==5))
                sample[itComp] = randGen->GetNormalVariate(100 * label, 1);
            else if ((itComp==16) && (label==6))
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

    // Save model and load inn new classifier
    const std::string outputModelFileName = "model.txt";
    const std::string outputSelectionModelFileName = "selection_model.txt";
    GMMClassifier->Save(outputModelFileName);
    GMMType::Pointer GMMClassifier2 = GMMType::New();
    GMMClassifier2->Load(outputModelFileName);

    std::remove( outputModelFileName.c_str() );
    std::remove( outputSelectionModelFileName.c_str() );

    // Classify with new model
    int TP = 0;
    for (int i = 0; i < nbSamples; ++i)
    {
        if (GMMClassifier2->Predict(InputListSample->GetMeasurementVector(i)) == TargetListSample->GetMeasurementVector(i))
            TP++;
    }

    // Test if overall accuracy is high. It is expected to be equal to 1 with this toy example
    if ( ((float) TP / nbSamples) > 0.9)
    {
        std::cout << "Model save and load succesfully" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Overall accuracy unexpectedly low. Possible problem with save/load." << std::endl;
        return EXIT_FAILURE;
    }

}
