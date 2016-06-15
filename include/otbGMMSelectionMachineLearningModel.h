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
  typedef typename Superclass::RealType                     RealType;
  typedef typename Superclass::MatrixType                   MatrixType;
  typedef typename Superclass::VectorType                   VectorType;

  typedef itk::Statistics::Subsample< InputListSampleType > ClassSampleType;
  typedef typename ClassSampleType::Pointer                 ClassSamplePointer;

  /** Run-time type information (and related methods). */
  itkNewMacro(Self);
  itkTypeMacro(GMMSelectionMachineLearningModel, GMMMachineLearningModel);

  void AddInstanceToFold(std::vector<InstanceIdentifier> & fold, int start, int end);
  void UpdateProportion();

  void ForwardSelection(std::string criterion, int selectedVarNb, int nfold);
  void FloatingForwardSelection(std::string criterion, int selectedVarNb, int nfold);

  ClassSamplePointer GetClassSamples(int classId);

protected:
  /** Constructor */
  GMMSelectionMachineLearningModel();

  /** Destructor */
  virtual ~GMMSelectionMachineLearningModel();

  /** PrintSelf method */
  void PrintSelf(std::ostream& os, itk::Indent indent) const;

  /** Vector of size C of scalar (2*log proportion) for each class */
  std::vector<RealType> m_Logprop;

private:
  GMMSelectionMachineLearningModel(const Self &); //purposely not implemented
  void operator =(const Self&); //purposely not implemented

  /** Array containing id of test samples for each class for cross validation */
  std::vector<ClassSamplePointer> m_Fold;
};
} // end namespace otb

#ifndef OTB_MANUAL_INSTANTIATION
#include "otbGMMSelectionMachineLearningModel.txx"
#endif

#endif