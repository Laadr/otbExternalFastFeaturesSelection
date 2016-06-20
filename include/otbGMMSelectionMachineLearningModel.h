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

  std::vector<int> GetSelectedVar();

  void ExtractVector(const std::vector<int> & indexes, const VectorType& input, VectorType& ouput);
  void ExtractVectorToColMatrix(const std::vector<int> & indexes, const VectorType& input, MatrixType& ouput);
  void ExtractReducedColumn(const int colIndex, const std::vector<int> & indexesRow, const MatrixType& input, MatrixType& ouput);
  void ExtractSubSymmetricMatrix(const std::vector<int> & indexes, const MatrixType& input, MatrixType& ouput);
  void AddInstanceToFold(typename InputListSampleType::Pointer samples, std::vector<InstanceIdentifier> & fold, int start, int end);
  void UpdateProportion();

  void ComputeClassifRate(std::vector<RealType> & criterionVal, const std::string direction, std::vector<int> & variablesPool, const std::string criterion);
  void ComputeJM         (std::vector<RealType> & JM, const std::string direction, std::vector<int> & variablesPool);
  void ComputeDivKL      (std::vector<RealType> & criterionVal, const std::string direction, std::vector<int> & variablesPool);

  void Selection(std::string direction, std::string criterion, int selectedVarNb, int nfold);
  void ForwardSelection(std::string criterion, int selectedVarNb);
  void FloatingForwardSelection(std::string criterion, int selectedVarNb);

  /** Train the machine learning model */
  virtual void Train();

  /** Predict values using the model */
  virtual TargetSampleType Predict(const InputSampleType& input, ConfidenceValueType *quality=NULL) const;

  ClassSamplePointer GetClassSamples(int classId);

protected:
  /** Constructor */
  GMMSelectionMachineLearningModel();

  /** Destructor */
  virtual ~GMMSelectionMachineLearningModel();

  /** PrintSelf method */
  void PrintSelf(std::ostream& os, itk::Indent indent) const;

  /** Vector of selected variables */
  std::vector<int> m_SelectedVar;

  /** Vector of size C containing the mean vector of each class for the selected variables */
  std::vector<VectorType> m_SubMeans;

  /** Vector of size C of scalar (2*log proportion) for each class */
  std::vector<RealType> m_Logprop;

  /** Vector of model for cross-validation */
  std::vector<GMMSelectionMachineLearningModel<TInputValue, TTargetValue>::Pointer > m_SubmodelCv;

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