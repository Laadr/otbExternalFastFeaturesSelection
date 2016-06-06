#ifndef __otbGMMMachineLearningModel_h
#define __otbGMMMachineLearningModel_h

#include "itkMacro.h"
#include "itkLightObject.h"
#include "itkFixedArray.h"
#include "itkArray.h"
#include "otbMachineLearningModel.h"

#include "itkCovarianceSampleFilter.h"
#include "itkSampleClassifierFilter.h"

namespace otb
{
template <class TInputValue, class TTargetValue>
class ITK_EXPORT GMMMachineLearningModel
  : public MachineLearningModel <TInputValue, TTargetValue>
{

public:
  /** Standard class typedefs. */
  typedef GMMMachineLearningModel                         Self;
  typedef MachineLearningModel<TInputValue, TTargetValue> Superclass;
  typedef itk::SmartPointer<Self>                         Pointer;
  typedef itk::SmartPointer<const Self>                   ConstPointer;

  typedef typename Superclass::InputValueType             InputValueType;
  typedef typename Superclass::InputSampleType            InputSampleType;
  typedef typename Superclass::InputListSampleType        InputListSampleType;
  typedef typename Superclass::TargetValueType            TargetValueType;
  typedef typename Superclass::TargetSampleType           TargetSampleType;
  typedef typename Superclass::TargetListSampleType       TargetListSampleType;
  typedef typename Superclass::ConfidenceValueType        ConfidenceValueType;

  /** Types of the mean and the covariance calculator that will update
   *  this component's distribution parameters */
  typedef itk::Statistics::CovarianceSampleFilter< itk::Statistics::Subsample< InputListSampleType > > CovarianceEstimatorType;
  typedef typename CovarianceEstimatorType::MatrixType MatrixType;
  typedef typename MatrixType::ValueType MatrixValueType;
  typedef typename CovarianceEstimatorType::MeasurementVectorRealType MeanVectorType;

  /** Run-time type information (and related methods). */
  itkNewMacro(Self);
  itkTypeMacro(GMMMachineLearningModel, MachineLearningModel);

  /** Set m_tau and update m_lambdaQ and m_cstDecision */
  void SetTau(MatrixValueType tau);

  /** Compute de decomposition in eigenvalues and eigenvectors of a matrix */
  void Decomposition(MatrixType &inputMatrix, MatrixType &outputMatrix, itk::VariableLengthVector<MatrixValueType> &eigenValues);

  /** Train the machine learning model */
  virtual void Train();

  /** Predict values using the model */
  virtual TargetSampleType Predict(const InputSampleType& input, ConfidenceValueType *quality=NULL) const;

  /** Save the model to file */
  virtual void Save(const std::string & filename, const std::string & name="");

  /** Load the model from file */
  virtual void Load(const std::string & filename, const std::string & name="");

  /**\name Classification model file compatibility tests */
  //@{
  /** Is the input model file readable and compatible with the corresponding classifier ? */
  virtual bool CanReadFile(const std::string &);

  /** Is the input model file writable and compatible with the corresponding classifier ? */
  virtual bool CanWriteFile(const std::string &);
  //@}

  // itkSetObjectMacro(m_MapOfClasses, MapOfClasses, std::map<TargetValueType, int >);
  // itkGetObjectMacro(m_MapOfClasses, MapOfClasses, std::map<TargetValueType, int >);
  // itkSetObjectMacro(m_MapOfIndices, MapOfIndices, std::map<int, TargetValueType >);
  // itkGetObjectMacro(m_MapOfIndices, MapOfIndices, std::map<int, TargetValueType >);
  // itkSetObjectMacro(m_classNb, ClassNb, unsigned int);
  // itkGetObjectMacro(m_classNb, ClassNb, unsigned int);
  // itkSetObjectMacro(m_featNb, FeatNb, unsigned int);
  // itkGetObjectMacro(m_featNb, FeatNb, unsigned int);

  void AddMean(MeanVectorType vector);
  void AddCovMatrix(MatrixType covMatrix);
  void AddNbSpl(unsigned long n);
  void UpdateProportion();

protected:
  /** Constructor */
  GMMMachineLearningModel();

  /** Destructor */
  virtual ~GMMMachineLearningModel();

  /** PrintSelf method */
  void PrintSelf(std::ostream& os, itk::Indent indent) const;

private:
  GMMMachineLearningModel(const Self &); //purposely not implemented
  void operator =(const Self&); //purposely not implemented

  typename CovarianceEstimatorType::Pointer m_CovarianceEstimator;

  /** Matrix (Cxd) containing the mean vector of each class */
  std::vector<MeanVectorType> m_Means;

  /** Vector of covariance matrix (dxd) of each class */
  std::vector<MatrixType> m_Covariances;

  /** Map of classes */
  std::map<TargetValueType, int> m_MapOfClasses;
  std::map<int, TargetValueType> m_MapOfIndices;

  /** Number of class */
  unsigned int m_classNb;

  /** Number of features */
  unsigned int m_featNb;

  /** Vector containing the number of samples in each class */
  std::vector<unsigned long> m_NbSpl;

  /** Vector containing the proportion of samples in each class */
  std::vector<float> m_Proportion;

  /** Vector of size C containing eigenvalues of the covariance matrices */
  std::vector<itk::VariableLengthVector<MatrixValueType> > m_eigenValues;

  /** Vector of size C of eigenvectors matrix (dxd) of each class (each line is an eigenvector) */
  std::vector<MatrixType> m_Q;

  /** Vector of size C of matrix (dxd) eigenvalues^(-1/2) * Q.T for each class */
  std::vector<MatrixType> m_lambdaQ;

  /** Vector of size C of scalar (logdet cov - 2*log proportion) for each class */
  std::vector<MatrixValueType> m_cstDecision;

  /** Regularisation constant */
  MatrixValueType m_tau;

  /** Create one subset of samples for each class */
  typedef itk::Statistics::Subsample< InputListSampleType > ClassSampleType;
  std::vector< typename ClassSampleType::Pointer > m_classSamples;

};
} // end namespace otb

#ifndef OTB_MANUAL_INSTANTIATION
#include "otbGMMMachineLearningModel.txx"
#endif

#endif