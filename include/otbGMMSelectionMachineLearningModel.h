#ifndef __otbGMMSelectionMachineLearningModel_h
#define __otbGMMSelectionMachineLearningModel_h

#include <string>
#include "itkMacro.h"
#include "itkLightObject.h"
#include "itkFixedArray.h"
#include "itkArray.h"
#include "otbMachineLearningModel.h"
#include "otbGMMMachineLearningModel.h"

#include "itkCovarianceSampleFilter.h"
#include "itkSampleClassifierFilter.h"

namespace otb
{
template <class TInputValue, class TTargetValue>
class ITK_EXPORT GMMSelectionMachineLearningModel
  : public GMMMachineLearningModel <TInputValue, TTargetValue>
{

public:
  /** Standard class typedefs. */
  typedef GMMSelectionMachineLearningModel                    Self;
  typedef GMMMachineLearningModel<TInputValue, TTargetValue>  Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  typedef typename Superclass::InputValueType               InputValueType;
  typedef typename Superclass::InputSampleType              InputSampleType;
  typedef typename Superclass::InputListSampleType          InputListSampleType;
  typedef typename InputListSampleType::InstanceIdentifier  InstanceIdentifier;
  typedef typename Superclass::TargetValueType              TargetValueType;
  typedef typename Superclass::TargetSampleType             TargetSampleType;
  typedef typename Superclass::TargetListSampleType         TargetListSampleType;
  typedef typename Superclass::ConfidenceValueType          ConfidenceValueType;

  /** Types of the mean and the covariance calculator that will update
   *  this component's distribution parameters */
  typedef itk::Statistics::CovarianceSampleFilter< itk::Statistics::Subsample< InputListSampleType > > CovarianceEstimatorType;
  typedef typename CovarianceEstimatorType::MatrixType MatrixType;
  typedef typename MatrixType::ValueType MatrixValueType;
  typedef typename CovarianceEstimatorType::MeasurementVectorRealType MeanVectorType;

  /** Run-time type information (and related methods). */
  itkNewMacro(Self);
  itkTypeMacro(GMMSelectionMachineLearningModel, GMMMachineLearningModel);

  void AddInstanceToFold(std::vector<InstanceIdentifier> & fold, int start, int end);

  void ForwardSelection(std::string method, std::string criterion, int selestedVarNb, int nfold);
  void FloatingForwardSelection(std::string method, std::string criterion, int selestedVarNb, int nfold);

protected:
  /** Constructor */
  GMMSelectionMachineLearningModel();

  /** Destructor */
  virtual ~GMMSelectionMachineLearningModel();

  /** PrintSelf method */
  void PrintSelf(std::ostream& os, itk::Indent indent) const;

private:
  GMMSelectionMachineLearningModel(const Self &); //purposely not implemented
  void operator =(const Self&); //purposely not implemented

  typename CovarianceEstimatorType::Pointer m_CovarianceEstimator;

  /** Array containing id of test samples for cross validation */
  std::vector<InstanceIdentifier> m_fold;

};
} // end namespace otb

#ifndef OTB_MANUAL_INSTANTIATION
#include "otbGMMSelectionMachineLearningModel.txx"
#endif

#endif