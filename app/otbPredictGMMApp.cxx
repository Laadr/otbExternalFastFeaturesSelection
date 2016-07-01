#include "otbWrapperApplication.h"
#include "otbWrapperApplicationFactory.h"

#include "otbGMMMachineLearningModel.h"
#include "otbGMMSelectionMachineLearningModel.h"

#include "otbImageClassificationFilter.h"
#include "otbShiftScaleVectorImageFilter.h"
// Normalize the samples
#include "otbShiftScaleSampleListFilter.h"
// Statistic XML Reader
#include "otbStatisticsXMLFileReader.h"
// only need this filter as a dummy process object
#include "otbRGBAPixelConverter.h"


namespace otb
{
namespace Wrapper
{

class PredictGMMApp : public Application
{
public:
  typedef PredictGMMApp Self;
  typedef itk::SmartPointer<Self> Pointer;

  /** Filters typedef */
  typedef UInt16ImageType                                                                      OutputImageType;
  typedef UInt8ImageType                                                                       MaskImageType;
  typedef itk::VariableLengthVector<FloatVectorImageType::InternalPixelType>                   MeasurementType;
  typedef StatisticsXMLFileReader<MeasurementType>                                             StatisticsReader;
  typedef ShiftScaleVectorImageFilter<FloatVectorImageType, FloatVectorImageType>              RescalerType;
  typedef ImageClassificationFilter<FloatVectorImageType, OutputImageType, MaskImageType>      ClassificationFilterType;
  typedef ClassificationFilterType::Pointer                                                    ClassificationFilterPointerType;
  typedef ClassificationFilterType::ValueType                                                  ValueType;
  typedef ClassificationFilterType::LabelType                                                  LabelType;
  typedef ClassificationFilterType::ConfidenceImageType                                        ConfidenceImageType;

  itkNewMacro(Self);
  itkTypeMacro(PredictGMMApp, Application);


private:

  ClassificationFilterType::Pointer m_ClassificationFilter;
  RescalerType::Pointer m_Rescaler;

  void DoInit()
  {
    SetName("PredictGMMApp");
    SetDescription("Performs a classification of the input image according to a GMM model file.");

    // Documentation
    SetDocName("Image Classification");
    SetDocLongDescription("This application performs an image classification based on a GMM model file produced by the TrainGMMApp or TrainGMMSelectionApp application. Pixels of the output image will contain the class labels decided by the classifier (maximal class label = 65535). The input pixels can be optionally centered and reduced according to the statistics file produced by the ComputeImagesStatistics application. An optional input mask can be provided, in which case only input image pixels whose corresponding mask value is greater than 0 will be classified. The remaining of pixels will be given the label 0 in the output image.");

    SetDocLimitations("The input image must have the same type, order and number of bands than the images used to produce the statistics file and the SVM model file. If a statistics file was used during training by the TrainImagesClassifier, it is mandatory to use the same statistics file for classification. If an input mask is used, its size must match the input image size.");
    SetDocAuthors("OTB-Team");
    SetDocSeeAlso("TrainGMMApp, TrainGMMSelectionApp, ComputeImagesStatistics");

    AddDocTag(Tags::Learning);

    AddParameter(ParameterType_InputImage, "in",  "Input Image");
    SetParameterDescription( "in", "The input image to classify.");

    AddParameter(ParameterType_InputImage,  "mask",   "Input Mask");
    SetParameterDescription( "mask", "The mask allows restricting classification of the input image to the area where mask pixel values are greater than 0.");
    MandatoryOff("mask");

    AddParameter(ParameterType_InputFilename, "model", "Model file");
    SetParameterDescription("model", "A model file (produced by TrainGMMApp or TrainGMMSelectionApp application, maximal class label = 65535).");

    AddParameter(ParameterType_Choice, "modeltype", "Type of GMM model");
    AddChoice("modeltype.basic", "Basic GMM possibly with Ridge regularization");
    AddChoice("modeltype.selection", "GMM with features selection");
    SetParameterDescription("modeltype", "Type of trained GMM type. If model is trained with TrainGMMApp, use basic and, if trained with TrainGMMSelectionApp, used selection.");


    AddParameter(ParameterType_InputFilename, "imstat", "Statistics file");
    SetParameterDescription("imstat", "A XML file containing mean and standard deviation to center and reduce samples before classification (produced by ComputeImagesStatistics application).");
    MandatoryOff("imstat");

    AddParameter(ParameterType_OutputImage, "out",  "Output Image");
    SetParameterDescription( "out", "Output image containing class labels");
    SetDefaultOutputPixelType( "out", ImagePixelType_uint8);

    AddParameter(ParameterType_OutputImage, "confmap",  "Confidence map");
    SetParameterDescription( "confmap", "Confidence map of the produced classification. The confidence index is the probability of the class given the sample.");
    SetDefaultOutputPixelType( "confmap", ImagePixelType_double);
    MandatoryOff("confmap");

    AddRAMParameter();

   // Doc example parameter settings
    SetDocExampleParameterValue("in", "QB_1_ortho.tif");
    SetDocExampleParameterValue("imstat", "EstimateImageStatisticsQB1.xml");
    SetDocExampleParameterValue("model", "model.txt");
    SetDocExampleParameterValue("modeltype", "basic");
    SetDocExampleParameterValue("out", "clLabeledImageQB1.tif");
  }

  void DoUpdateParameters()
  {
  }

  void DoExecute()
  {
    // Load input image
    FloatVectorImageType::Pointer inImage = GetParameterImage("in");
    inImage->UpdateOutputInformation();

    // Load model
    otbAppLogINFO("Loading model");

    typedef MachineLearningModel<ValueType, LabelType> ModelType;
    ModelType::Pointer GMMClassifier;
    if (GetParameterString("modeltype").compare("basic") == 0)
      GMMClassifier = GMMMachineLearningModel<ValueType, LabelType>::New();
    else //if (GetParameterString("modeltype").compare("selection") == 0)
      GMMClassifier = GMMSelectionMachineLearningModel<ValueType, LabelType>::New();

    GMMClassifier->Load(GetParameterString("model"));
    otbAppLogINFO("Model loaded");

    // Normalize input image (optional)
    StatisticsReader::Pointer  statisticsReader = StatisticsReader::New();
    MeasurementType  meanMeasurementVector;
    MeasurementType  stddevMeasurementVector;
    m_Rescaler = RescalerType::New();

    // Classify
    m_ClassificationFilter = ClassificationFilterType::New();
    m_ClassificationFilter->SetModel(GMMClassifier);

    // Normalize input image if asked
    if(IsParameterEnabled("imstat"))
    {
      otbAppLogINFO("Input image normalization activated.");
      // Load input image statistics
      statisticsReader->SetFileName(GetParameterString("imstat"));
      meanMeasurementVector   = statisticsReader->GetStatisticVectorByName("mean");
      stddevMeasurementVector = statisticsReader->GetStatisticVectorByName("stddev");
      otbAppLogINFO( "mean used: " << meanMeasurementVector );
      otbAppLogINFO( "standard deviation used: " << stddevMeasurementVector );
      // Rescale vector image
      m_Rescaler->SetScale(stddevMeasurementVector);
      m_Rescaler->SetShift(meanMeasurementVector);
      m_Rescaler->SetInput(inImage);

      m_ClassificationFilter->SetInput(m_Rescaler->GetOutput());
    }
    else
    {
      otbAppLogINFO("Input image normalization deactivated.");
      m_ClassificationFilter->SetInput(inImage);
    }


    if(IsParameterEnabled("mask"))
    {
      otbAppLogINFO("Using input mask");
      // Load mask image and cast into LabeledImageType
      MaskImageType::Pointer inMask = GetParameterUInt8Image("mask");

      m_ClassificationFilter->SetInputMask(inMask);
    }

    SetParameterOutputImage<OutputImageType>("out", m_ClassificationFilter->GetOutput());

    // output confidence map
    if (IsParameterEnabled("confmap") && HasValue("confmap"))
    {
      std::cout << HasValue("confmap") << std::endl;
      m_ClassificationFilter->SetUseConfidenceMap(true);
      if (GMMClassifier->HasConfidenceIndex())
      {
        SetParameterOutputImage<ConfidenceImageType>("confmap",m_ClassificationFilter->GetOutputConfidence());
      }
      else
      {
        otbAppLogWARNING("Confidence map requested but the classifier doesn't support it!");
        this->DisableParameter("confmap");
      }
    }
  }

};

} // end namespace Wrapper
} // end namespace otb

OTB_APPLICATION_EXPORT(otb::Wrapper::PredictGMMApp)
