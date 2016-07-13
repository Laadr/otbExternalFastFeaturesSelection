#include "otbWrapperApplication.h"
#include "otbWrapperApplicationFactory.h"

#include "otbGMMMachineLearningModel.h"
#include "otbGMMSelectionMachineLearningModel.h"

#include "otbListSampleGenerator.h"
// Validation
#include "otbConfusionMatrixCalculator.h"
// List sample concatenation
#include "otbConcatenateSampleListFilter.h"
// Normalize the samples
#include "otbShiftScaleSampleListFilter.h"
// Extract a ROI of the vectordata
#include "otbVectorDataIntoImageProjectionFilter.h"
// Statistic XML Reader
#include "otbStatisticsXMLFileReader.h"
// only need this filter as a dummy process object
#include "otbRGBAPixelConverter.h"


namespace otb
{
namespace Wrapper
{

class TrainGMMSelectionApp : public Application
{
public:
  typedef TrainGMMSelectionApp Self;
  typedef itk::SmartPointer<Self> Pointer;

  typedef float    InputValueType;
  typedef int      TargetValueType;
  typedef GMMSelectionMachineLearningModel<InputValueType, TargetValueType> GMMType;


  typedef GMMType::InputSampleType         SampleType;
  typedef GMMType::InputListSampleType     ListSampleType;
  typedef GMMType::TargetSampleType        TargetSampleType;
  typedef GMMType::TargetListSampleType    TargetListSampleType;

  typedef VectorImage<InputValueType>         SampleImageType;
  typedef SampleImageType::PixelType          PixelType;

  // SampleList manipulation
  typedef ListSampleGenerator<SampleImageType, VectorDataType> ListSampleGeneratorType;
  typedef Statistics::ConcatenateSampleListFilter<ListSampleType> ConcatenateListSampleFilterType;
  typedef Statistics::ConcatenateSampleListFilter<TargetListSampleType> ConcatenateLabelListSampleFilterType;

  // Enhance List Sample  typedef otb::Statistics::ListSampleToBalancedListSampleFilter<ListSampleType, LabelListSampleType>      BalancingListSampleFilterType;
  typedef Statistics::ShiftScaleSampleListFilter<ListSampleType, ListSampleType> ShiftScaleFilterType;

  // Statistic XML file Reader
  typedef StatisticsXMLFileReader<SampleType> StatisticsReader;

  // Estimate performance on validation sample
  typedef ConfusionMatrixCalculator<TargetListSampleType, TargetListSampleType> ConfusionMatrixCalculatorType;
  typedef ConfusionMatrixCalculatorType::ConfusionMatrixType ConfusionMatrixType;
  typedef ConfusionMatrixCalculatorType::MapOfIndicesType MapOfIndicesType;
  typedef ConfusionMatrixCalculatorType::ClassLabelType ClassLabelType;

  // VectorData projection filter
  typedef VectorDataProjectionFilter<VectorDataType, VectorDataType> VectorDataProjectionFilterType;

  // Extract ROI
  typedef VectorDataIntoImageProjectionFilter<VectorDataType, SampleImageType> VectorDataReprojectionType;

  itkNewMacro(Self);
  itkTypeMacro(TrainGMMSelectionApp, Application);


private:

  GMMType::Pointer GMMClassifier;

  void DoInit()
  {
    SetName("TrainGMMSelectionApp");
    SetDescription("Train a GMM classifier from multiple pairs of images and training vector data.");

    // Documentation
    SetDocName("Train a GMM classifier from multiple images");
    SetDocLongDescription(
      "This application performs a GMM classifier training from multiple pairs of input images and training vector data. "
      "Samples are composed of pixel values in each band optionally centered and possibly reduced using an XML statistics file produced by "
      "the ComputeImagesStatistics application.\n The training vector data must contain polygons with a positive integer field "
      "representing the class label. The name of this field can be set using the \"Class label field\" parameter. Training and validation "
      "sample lists are built such that each class is equally represented in both lists. One parameter allows controlling the ratio "
      "between the number of samples in training and validation sets. Two parameters allow managing the size of the training and "
      "validation sets per class and per image.\n In addition, a features selection method is used during training to reduce dimensionality of the data.\n"
      " In the validation process, the confusion matrix is organized the following way: rows = reference labels, columns = produced labels. "
      "In the header of the optional confusion matrix output file, the validation (reference) and predicted (produced) class labels"
      " are ordered according to the rows/columns of the confusion matrix.\n This application is part of an external module developed by "
      "Adrien Lagrange (ad.lagrange@gmail.com) and Mathieu Fauvel.");
    SetDocLimitations("None");
    SetDocAuthors("Adrien Lagrange");
    SetDocSeeAlso("See also git depository https://github.com/Laadr/otbExternalFastFeaturesSelection.");

    AddDocTag(Tags::Learning);

    //Group IO
    AddParameter(ParameterType_Group, "io", "Input and output data");
    SetParameterDescription("io", "This group of parameters allows setting input and output data.");
    AddParameter(ParameterType_InputImageList, "io.il", "Input Image List");
    SetParameterDescription("io.il", "A list of input images.");
    AddParameter(ParameterType_InputVectorDataList, "io.vd", "Input Vector Data List");
    SetParameterDescription("io.vd", "A list of vector data to select the training samples.");
    AddParameter(ParameterType_InputFilename, "io.imstat", "Input XML image statistics file");
    MandatoryOff("io.imstat");
    SetParameterDescription("io.imstat",
                            "Input XML file containing the mean and the standard deviation of the input images.");
    AddParameter(ParameterType_OutputFilename, "io.confmatout", "Output confusion matrix");
    SetParameterDescription("io.confmatout", "Output file containing the confusion matrix (.csv format).");
    MandatoryOff("io.confmatout");
    AddParameter(ParameterType_OutputFilename, "io.out", "Output model");
    SetParameterDescription("io.out", "Output file containing the model estimated (.txt format).");


    //Group Sample list
    AddParameter(ParameterType_Group, "sample", "Training and validation samples parameters");
    SetParameterDescription("sample",
                            "This group of parameters allows you to set training and validation sample lists parameters.");

    AddParameter(ParameterType_Int, "sample.mt", "Maximum training sample size per class");
    SetDefaultParameterInt("sample.mt", 1000);
    SetParameterDescription("sample.mt", "Maximum size per class (in pixels) of the training sample list (default = 1000) (no limit = -1). If equal to -1, then the maximal size of the available training sample list per class will be equal to the surface area of the smallest class multiplied by the training sample ratio.");

    AddParameter(ParameterType_Int, "sample.mv", "Maximum validation sample size per class");
    SetDefaultParameterInt("sample.mv", 1000);
    SetParameterDescription("sample.mv", "Maximum size per class (in pixels) of the validation sample list (default = 1000) (no limit = -1). If equal to -1, then the maximal size of the available validation sample list per class will be equal to the surface area of the smallest class multiplied by the validation sample ratio.");

    AddParameter(ParameterType_Int, "sample.bm", "Bound sample number by minimum");
    SetDefaultParameterInt("sample.bm", 1);
    SetParameterDescription("sample.bm", "Bound the number of samples for each class by the number of available samples by the smallest class. Proportions between training and validation are respected. Default is true (=1).");

    AddParameter(ParameterType_Empty, "sample.edg", "On edge pixel inclusion");
    SetParameterDescription("sample.edg",
                            "Takes pixels on polygon edge into consideration when building training and validation samples.");
    MandatoryOff("sample.edg");

    AddParameter(ParameterType_Float, "sample.vtr", "Training and validation sample ratio");
    SetParameterDescription("sample.vtr",
                            "Ratio between training and validation samples (0.0 = all training, 1.0 = all validation) (default = 0.5).");
    SetParameterFloat("sample.vtr", 0.5);

    AddParameter(ParameterType_String, "sample.vfn", "Name of the discrimination field");
    SetParameterDescription("sample.vfn", "Name of the field used to discriminate class labels in the input vector data files.");
    SetParameterString("sample.vfn", "Class");


    //Group classifier parameters
    AddParameter(ParameterType_Group, "gmm", "Paramaters of GMM classifier.");
    SetParameterDescription("gmm", "This group of parameters allows to set the parameters of the GMM learning algorithm.");

    AddParameter(ParameterType_Int, "gmm.varnb", "Number of variables to select");
    SetParameterDescription("gmm.varnb", "Number of variables to select with the selection algorithm. All selected are not necessarily used for prediction depending of gmm.best option. ");

    AddParameter(ParameterType_Choice, "gmm.method", "Method used for selection");
    AddChoice("gmm.method.forward", "Sequential forward selection");
    AddChoice("gmm.method.sffs", "Sequential floating forward selection");
    SetParameterString("gmm.method", "forward");
    SetParameterDescription("gmm.method", "Method used for selection (default = forward). Two methods are possible the forward and the floating forward (forward/sffs).");

    AddParameter(ParameterType_Choice, "gmm.crit", "Criterion function for selection");
    AddChoice("gmm.crit.jm", "Jeffries-Matusita distance");
    AddChoice("gmm.crit.divkl", "Kullback–Leibler divergence");
    AddChoice("gmm.crit.accuracy", "Overall Accuracy");
    AddChoice("gmm.crit.kappa", "Cohen's kappa");
    AddChoice("gmm.crit.f1mean", "Mean of F1-scores");
    SetParameterString("gmm.crit", "jm");
    SetParameterDescription("gmm.crit", "Criterion function to use for features selection (default = jm). The five criterion functions available are Jeffries-Matusita distance, Kullback–Leibler divergence, overall accuracy, Cohen's kappa and mean of F1-scores (jm/divkl/accuracy/kappa/f1mean).");

    AddParameter(ParameterType_Int, "gmm.ncv", "Number of folds for cross-validation");
    SetDefaultParameterInt("gmm.ncv", 5);
    SetParameterDescription("gmm.ncv", "Number of folds for cross-validation to estimate the classification rate when selecting variables (default = 5). It is ussed only with Jeffries-Matusita distance and Kullback–Leibler divergence.");

    AddParameter(ParameterType_Int, "gmm.best", "If 1, choose optimal set of features based on criterion function after selection. If 0, all selected features are used");
    SetDefaultParameterInt("gmm.best", 1);
    SetParameterDescription("gmm.best","If 1, choose optimal set of features based on criterion function after selection (default). If 0, all selected features are used.");

    AddParameter(ParameterType_Int, "gmm.seed", "Rand seed for cross-validation");
    SetParameterInt("gmm.seed", 0);
    SetParameterDescription("gmm.seed", "Rand seed for cross-validation.");

    // Doc example parameter settings
    SetDocExampleParameterValue("io.il", "QB_1_ortho.tif");
    SetDocExampleParameterValue("io.vd", "VectorData_QB1.shp");
    SetDocExampleParameterValue("io.imstat", "EstimateImageStatisticsQB1.xml");
    SetDocExampleParameterValue("sample.mv", "100");
    SetDocExampleParameterValue("sample.mt", "100");
    SetDocExampleParameterValue("sample.vtr", "0.5");
    SetDocExampleParameterValue("sample.edg", "false");
    SetDocExampleParameterValue("sample.vfn", "Class");
    SetDocExampleParameterValue("gmm.varnb", "20");
    SetDocExampleParameterValue("gmm.method", "forward");
    SetDocExampleParameterValue("gmm.crit", "jm");
    SetDocExampleParameterValue("gmm.ncv", "5");
    SetDocExampleParameterValue("gmm.best", "1");
    SetDocExampleParameterValue("gmm.seed", "0");
    SetDocExampleParameterValue("io.out", "svmModelQB1.txt");
    SetDocExampleParameterValue("io.confmatout", "svmConfusionMatrixQB1.csv");
  }

  void DoUpdateParameters()
  {
  }

  void LogConfusionMatrix(ConfusionMatrixCalculatorType* confMatCalc)
  {
    ConfusionMatrixCalculatorType::ConfusionMatrixType matrix = confMatCalc->GetConfusionMatrix();

    // Compute minimal width
    size_t minwidth = 0;

    for (unsigned int i = 0; i < matrix.Rows(); i++)
    {
      for (unsigned int j = 0; j < matrix.Cols(); j++)
      {
        std::ostringstream os;
        os << matrix(i, j);
        size_t size = os.str().size();

        if (size > minwidth)
        {
          minwidth = size;
        }
      }
    }

    MapOfIndicesType mapOfIndices = confMatCalc->GetMapOfIndices();

    MapOfIndicesType::const_iterator it = mapOfIndices.begin();
    MapOfIndicesType::const_iterator end = mapOfIndices.end();

    for (; it != end; ++it)
    {
      std::ostringstream os;
      os << "[" << it->second << "]";

      size_t size = os.str().size();
      if (size > minwidth)
      {
        minwidth = size;
      }
    }

    // Generate matrix string, with 'minwidth' as size specifier
    std::ostringstream os;

    // Header line
    for (size_t i = 0; i < minwidth; ++i)
      os << " ";
    os << " ";

    it = mapOfIndices.begin();
    end = mapOfIndices.end();
    for (; it != end; ++it)
    {
      os << "[" << it->second << "]" << " ";
    }

    os << std::endl;

    // Each line of confusion matrix
    for (unsigned int i = 0; i < matrix.Rows(); i++)
    {
      ConfusionMatrixCalculatorType::ClassLabelType label = mapOfIndices[i];
      os << "[" << std::setw(minwidth - 2) << label << "]" << " ";
      for (unsigned int j = 0; j < matrix.Cols(); j++)
      {
        os << std::setw(minwidth) << matrix(i, j) << " ";
      }
      os << std::endl;
    }

    otbAppLogINFO("Confusion matrix (rows = reference labels, columns = produced labels):\n" << os.str());
  }

  void DoExecute()
  {
    //Create training and validation for list samples and label list samples
    ConcatenateLabelListSampleFilterType::Pointer concatenateTrainingLabels   = ConcatenateLabelListSampleFilterType::New();
    ConcatenateListSampleFilterType::Pointer concatenateTrainingSamples       = ConcatenateListSampleFilterType::New();
    ConcatenateLabelListSampleFilterType::Pointer concatenateValidationLabels = ConcatenateLabelListSampleFilterType::New();
    ConcatenateListSampleFilterType::Pointer concatenateValidationSamples     = ConcatenateListSampleFilterType::New();

    SampleType meanMeasurementVector;
    SampleType stddevMeasurementVector;

    //--------------------------
    // Load measurements from images
    unsigned int nbBands = 0;
    //Iterate over all input images

    FloatVectorImageListType* imageList = GetParameterImageList("io.il");
    VectorDataListType* vectorDataList  = GetParameterVectorDataList("io.vd");

    VectorDataReprojectionType::Pointer vdreproj;
    vdreproj = VectorDataReprojectionType::New();

    //Iterate over all input images
    for (unsigned int imgIndex = 0; imgIndex < imageList->Size(); ++imgIndex)
    {
      std::ostringstream oss1, oss2;
      oss1 << "Reproject polygons for image " << (imgIndex+1) << " ...";
      oss2 << "Extract samples from image " << (imgIndex+1) << " ...";

      FloatVectorImageType::Pointer image = imageList->GetNthElement(imgIndex);
      image->UpdateOutputInformation();

      if (imgIndex == 0)
      {
        nbBands = image->GetNumberOfComponentsPerPixel();
      }

      // read the Vectordata
      vdreproj->SetInputImage(image);
      vdreproj->SetInput(vectorDataList->GetNthElement(imgIndex));
      vdreproj->SetUseOutputSpacingAndOriginFromImage(false);

      AddProcess(vdreproj, oss1.str());
      vdreproj->Update();

      //Sample list generator
      ListSampleGeneratorType::Pointer sampleGenerator = ListSampleGeneratorType::New();

      sampleGenerator->SetInput(image);
      sampleGenerator->SetInputVectorData(vdreproj->GetOutput());

      sampleGenerator->SetClassKey(GetParameterString("sample.vfn"));
      sampleGenerator->SetMaxTrainingSize(GetParameterInt("sample.mt"));
      sampleGenerator->SetMaxValidationSize(GetParameterInt("sample.mv"));
      sampleGenerator->SetValidationTrainingProportion(GetParameterFloat("sample.vtr"));
      sampleGenerator->SetBoundByMin(GetParameterInt("sample.bm")!=0);

      // take pixel located on polygon edge into consideration
      if (IsParameterEnabled("sample.edg"))
      {
        sampleGenerator->SetPolygonEdgeInclusion(true);
      }

      AddProcess(sampleGenerator, oss2.str());
      sampleGenerator->Update();

      TargetListSampleType::Pointer trainLabels = sampleGenerator->GetTrainingListLabel();
      ListSampleType::Pointer trainSamples      = sampleGenerator->GetTrainingListSample();
      TargetListSampleType::Pointer validLabels = sampleGenerator->GetValidationListLabel();
      ListSampleType::Pointer validSamples      = sampleGenerator->GetValidationListSample();

      trainLabels->DisconnectPipeline();
      trainSamples->DisconnectPipeline();
      validLabels->DisconnectPipeline();
      validSamples->DisconnectPipeline();

      //Concatenate training and validation samples from the image
      concatenateTrainingLabels->AddInput(trainLabels);
      concatenateTrainingSamples->AddInput(trainSamples);
      concatenateValidationLabels->AddInput(validLabels);
      concatenateValidationSamples->AddInput(validSamples);
    }

    // Update
    AddProcess(concatenateValidationLabels, "Concatenate samples ...");
    concatenateTrainingSamples->Update();
    concatenateTrainingLabels->Update();
    concatenateValidationSamples->Update();
    concatenateValidationLabels->Update();

    if (concatenateTrainingSamples->GetOutput()->Size() == 0)
    {
      otbAppLogFATAL("No training samples, cannot perform training.");
    }

    if (concatenateValidationSamples->GetOutput()->Size() == 0)
    {
      otbAppLogWARNING("No validation samples.");
    }

    if (IsParameterEnabled("io.imstat"))
    {
      StatisticsReader::Pointer statisticsReader = StatisticsReader::New();
      statisticsReader->SetFileName(GetParameterString("io.imstat"));
      meanMeasurementVector = statisticsReader->GetStatisticVectorByName("mean");
      stddevMeasurementVector = statisticsReader->GetStatisticVectorByName("stddev");
    }
    else
    {
      meanMeasurementVector.SetSize(nbBands);
      meanMeasurementVector.Fill(0.);
      stddevMeasurementVector.SetSize(nbBands);
      stddevMeasurementVector.Fill(1.);
    }

    // Shift scale the training samples
    ShiftScaleFilterType::Pointer trainingShiftScaleFilter = ShiftScaleFilterType::New();
    trainingShiftScaleFilter->SetInput(concatenateTrainingSamples->GetOutput());
    trainingShiftScaleFilter->SetShifts(meanMeasurementVector);
    trainingShiftScaleFilter->SetScales(stddevMeasurementVector);
    AddProcess(trainingShiftScaleFilter, "Normalize training samples ...");
    trainingShiftScaleFilter->Update();

    ListSampleType::Pointer validationListSample=ListSampleType::New();

    // Shift scale the validation samples
    if ( concatenateValidationSamples->GetOutput()->Size() != 0 )
    {
      ShiftScaleFilterType::Pointer validationShiftScaleFilter = ShiftScaleFilterType::New();
      validationShiftScaleFilter->SetInput(concatenateValidationSamples->GetOutput());
      validationShiftScaleFilter->SetShifts(meanMeasurementVector);
      validationShiftScaleFilter->SetScales(stddevMeasurementVector);
      AddProcess(validationShiftScaleFilter, "Normalize validation samples ...");
      validationShiftScaleFilter->Update();
      validationListSample = validationShiftScaleFilter->GetOutput();
    }

    ListSampleType::Pointer trainingListSample;
    TargetListSampleType::Pointer trainingLabeledListSample;

    trainingListSample = trainingShiftScaleFilter->GetOutput();
    trainingLabeledListSample = concatenateTrainingLabels->GetOutput();
    otbAppLogINFO("Number of training samples: " << concatenateTrainingSamples->GetOutput()->Size());

    TargetListSampleType::Pointer validationLabeledListSample = concatenateValidationLabels->GetOutput();
    otbAppLogINFO("Size of training set: " << trainingListSample->Size());
    otbAppLogINFO("Size of validation set: " << validationListSample->Size());
    otbAppLogINFO("Size of labeled training set: " << trainingLabeledListSample->Size());
    otbAppLogINFO("Size of labeled validation set: " << validationLabeledListSample->Size());

    // Training
    GMMClassifier = GMMType::New();

    GMMClassifier->SetInputListSample(trainingListSample);
    GMMClassifier->SetTargetListSample(trainingLabeledListSample);

    // Setup fake reporter
    RGBAPixelConverter<int,int>::Pointer dummyFilter = RGBAPixelConverter<int,int>::New();
    dummyFilter->SetProgress(0.0f);
    this->AddProcess(dummyFilter,"Train...");
    dummyFilter->InvokeEvent(itk::StartEvent());

    GMMClassifier->Train();

    GMMClassifier->SetEnableOptimalSet((bool) GetParameterInt("gmm.best"));
    GMMClassifier->Selection(GetParameterString("gmm.method"),GetParameterString("gmm.crit"),GetParameterInt("gmm.varnb"),GetParameterInt("gmm.ncv"),GetParameterInt("gmm.seed"));

    // update reporter
    dummyFilter->UpdateProgress(1.0f);
    dummyFilter->InvokeEvent(itk::EndEvent());

    GMMClassifier->Save(GetParameterString("io.out"));

    // Test of performance

    TargetListSampleType::Pointer predictedList                = TargetListSampleType::New();
    ListSampleType::Pointer performanceListSample              = ListSampleType::New();
    TargetListSampleType::Pointer performanceLabeledListSample = TargetListSampleType::New();

    //Test the input validation set size
    if(validationLabeledListSample->Size() != 0)
    {
      performanceListSample = validationListSample;
      performanceLabeledListSample = validationLabeledListSample;
    }
    else
    {
      otbAppLogWARNING("The validation set is empty. The performance estimation is done using the input training set in this case.");
      performanceListSample = trainingListSample;
      performanceLabeledListSample = trainingLabeledListSample;
    }

    // Setup fake reporter
    dummyFilter->SetProgress(0.0f);
    this->AddProcess(dummyFilter,"Classify...");
    dummyFilter->InvokeEvent(itk::StartEvent());

    for(typename ListSampleType::ConstIterator sIt = performanceListSample->Begin(); sIt != performanceListSample->End(); ++sIt)
      predictedList->PushBack( GMMClassifier->Predict(sIt.GetMeasurementVector()) );

    // update reporter
    dummyFilter->UpdateProgress(1.0f);
    dummyFilter->InvokeEvent(itk::EndEvent());

    ConfusionMatrixCalculatorType::Pointer confMatCalc = ConfusionMatrixCalculatorType::New();

    otbAppLogINFO("Predicted list size : " << predictedList->Size());
    otbAppLogINFO("ValidationLabeledListSample size : " << performanceLabeledListSample->Size());
    confMatCalc->SetReferenceLabels(performanceLabeledListSample);
    confMatCalc->SetProducedLabels(predictedList);
    confMatCalc->Compute();

    otbAppLogINFO("training performances");
    LogConfusionMatrix(confMatCalc);

    for (unsigned int itClasses = 0; itClasses < confMatCalc->GetNumberOfClasses(); itClasses++)
    {
      ConfusionMatrixCalculatorType::ClassLabelType classLabel = confMatCalc->GetMapOfIndices()[itClasses];

      otbAppLogINFO("Precision of class [" << classLabel << "] vs all: " << confMatCalc->GetPrecisions()[itClasses]);
      otbAppLogINFO("Recall of class    [" << classLabel << "] vs all: " << confMatCalc->GetRecalls()[itClasses]);
      otbAppLogINFO(
        "F-score of class   [" << classLabel << "] vs all: " << confMatCalc->GetFScores()[itClasses] << "\n");
    }
    otbAppLogINFO("Global performance, Kappa index: " << confMatCalc->GetKappaIndex());


    if (this->HasValue("io.confmatout"))
    {
      // Writing the confusion matrix in the output .CSV file

      MapOfIndicesType::iterator itMapOfIndicesValid, itMapOfIndicesPred;
      ClassLabelType labelValid = 0;

      ConfusionMatrixType confusionMatrix = confMatCalc->GetConfusionMatrix();
      MapOfIndicesType mapOfIndicesValid = confMatCalc->GetMapOfIndices();

      unsigned int nbClassesPred = mapOfIndicesValid.size();

      /////////////////////////////////////////////
      // Filling the 2 headers for the output file
      const std::string commentValidStr = "#Reference labels (rows):";
      const std::string commentPredStr = "#Produced labels (columns):";
      const char separatorChar = ',';
      std::ostringstream ossHeaderValidLabels, ossHeaderPredLabels;

      // Filling ossHeaderValidLabels and ossHeaderPredLabels for the output file
      ossHeaderValidLabels << commentValidStr;
      ossHeaderPredLabels << commentPredStr;

      itMapOfIndicesValid = mapOfIndicesValid.begin();

      while (itMapOfIndicesValid != mapOfIndicesValid.end())
      {
        // labels labelValid of mapOfIndicesValid are already sorted in otbConfusionMatrixCalculator
        labelValid = itMapOfIndicesValid->second;

        otbAppLogINFO("mapOfIndicesValid[" << itMapOfIndicesValid->first << "] = " << labelValid);

        ossHeaderValidLabels << labelValid;
        ossHeaderPredLabels << labelValid;

        ++itMapOfIndicesValid;

        if (itMapOfIndicesValid != mapOfIndicesValid.end())
        {
          ossHeaderValidLabels << separatorChar;
          ossHeaderPredLabels << separatorChar;
        }
        else
        {
          ossHeaderValidLabels << std::endl;
          ossHeaderPredLabels << std::endl;
        }
      }

      std::ofstream outFile;
      outFile.open(this->GetParameterString("io.confmatout").c_str());
      outFile << std::fixed;
      outFile.precision(10);

      /////////////////////////////////////
      // Writing the 2 headers
      outFile << ossHeaderValidLabels.str();
      outFile << ossHeaderPredLabels.str();
      /////////////////////////////////////

      unsigned int indexLabelValid = 0, indexLabelPred = 0;

      for (itMapOfIndicesValid = mapOfIndicesValid.begin(); itMapOfIndicesValid != mapOfIndicesValid.end(); ++itMapOfIndicesValid)
      {
        indexLabelPred = 0;

        for (itMapOfIndicesPred = mapOfIndicesValid.begin(); itMapOfIndicesPred != mapOfIndicesValid.end(); ++itMapOfIndicesPred)
        {
          // Writing the confusion matrix (sorted in otbConfusionMatrixCalculator) in the output file
          outFile << confusionMatrix(indexLabelValid, indexLabelPred);
          if (indexLabelPred < (nbClassesPred - 1))
          {
            outFile << separatorChar;
          }
          else
          {
            outFile << std::endl;
          }
          ++indexLabelPred;
        }

        ++indexLabelValid;
      }

      outFile.close();
    }
  }

};

} // end namespace Wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT(otb::Wrapper::TrainGMMSelectionApp)
