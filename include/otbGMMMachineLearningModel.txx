#ifndef __otbGMMMachineLearningModel_txx
#define __otbGMMMachineLearningModel_txx

#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include <vector>
#include "itkMacro.h"
#include "itkSubsample.h"
#include "otbGMMMachineLearningModel.h"
#include "otbOpenCVUtils.h"
#include "vnl/vnl_copy.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "vnl/algo/vnl_generalized_eigensystem.h"

namespace otb
{

template <class TInputValue, class TOutputValue>
GMMMachineLearningModel<TInputValue,TOutputValue>
::GMMMachineLearningModel():
  m_ClassNb(0),
  m_FeatNb(0),
  m_Tau(0)
{
}


template <class TInputValue, class TOutputValue>
GMMMachineLearningModel<TInputValue,TOutputValue>
::~GMMMachineLearningModel()
{
}

/** Set m_Tau and update m_LambdaQ and m_CstDecision */
template <class TInputValue, class TOutputValue>
void GMMMachineLearningModel<TInputValue,TOutputValue>
::SetTau(RealType tau)
{
  m_Tau = tau;

  // Precompute lambda^(-1/2) * Q and log(det lambda)
  if (!m_Q.empty())
  {
    RealType lambda;
    m_CstDecision.resize(m_ClassNb,0);
    m_LambdaQ.resize(m_ClassNb, MatrixType(m_FeatNb,m_FeatNb));

    for (int i = 0; i < m_ClassNb; ++i)
    {
      for (int j = 0; j < m_FeatNb; ++j)
      {
        lambda = 1 / sqrt(m_EigenValues[i][j] + m_Tau);
        for (int k = 0; k < m_FeatNb; ++k)
          m_LambdaQ[i](j,k) = lambda * m_Q[i](k,j);

        m_CstDecision[i] += log(m_EigenValues[i][j] + m_Tau);
      }

      m_CstDecision[i] += -2*log(m_Proportion[i]);
    }
  }
}

/** Compute de decomposition in eigenvalues and eigenvectors of a matrix */
template <class TInputValue, class TOutputValue>
void GMMMachineLearningModel<TInputValue,TOutputValue>
::Decomposition(MatrixType &inputMatrix, MatrixType &outputMatrix, VectorType &eigenValues)
{
  vnl_symmetric_eigensystem_compute( inputMatrix, outputMatrix, eigenValues );

  for (int i = 0; i < eigenValues.size(); ++i)
  {
    if (eigenValues[i] < std::numeric_limits<RealType>::epsilon())
      eigenValues[i] = std::numeric_limits<RealType>::epsilon();
  }
}

/** Train the machine learning model */
template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::Train()
{
  // Get pointer to samples and labels
  typename InputListSampleType::Pointer samples = this->GetInputListSample();
  typename TargetListSampleType::Pointer labels = this->GetTargetListSample();

  // Declare iterator for samples and labels
  typename TargetListSampleType::ConstIterator refIterator  = labels->Begin();
  typename InputListSampleType::ConstIterator inputIterator = samples->Begin();

  // Get number of samples
  unsigned long sampleNb = labels->Size();

  // Get number of features
  m_FeatNb = samples->GetMeasurementVectorSize();

  // Get number of classes and map indice with label
  TargetValueType currentLabel;
  while (refIterator != labels->End())
  {
    currentLabel = refIterator.GetMeasurementVector()[0];
    if (m_MapOfClasses.find(currentLabel) == m_MapOfClasses.end())
    {
      m_MapOfClasses[currentLabel] = m_ClassNb;
      m_MapOfIndices[m_ClassNb]    = currentLabel;
      ++m_ClassNb;
    }
    ++refIterator;
  }

  // Create one subsample set for each class
  m_ClassSamples.reserve(m_ClassNb);
  for ( unsigned int i = 0; i < m_ClassNb; ++i )
  {
    m_ClassSamples.push_back( ClassSampleType::New() );
    m_ClassSamples[i]->SetSample( samples );
  }
  inputIterator = samples->Begin();
  refIterator   = labels->Begin();
  while (inputIterator != samples->End())
  {
    currentLabel = refIterator.GetMeasurementVector()[0];
    m_ClassSamples[m_MapOfClasses[currentLabel]]->AddInstance( inputIterator.GetInstanceIdentifier() );
    ++inputIterator;
    ++refIterator;
  }

  // Allocate all members
  m_NbSpl.resize(m_ClassNb);
  m_Proportion.resize(m_ClassNb);
  m_Covariances.resize(m_ClassNb,MatrixType(m_FeatNb,m_FeatNb));
  m_Means.resize(m_ClassNb,VectorType(m_FeatNb));

  m_Q.resize(m_ClassNb,MatrixType(m_FeatNb,m_FeatNb));
  m_EigenValues.resize(m_ClassNb,VectorType(m_FeatNb));

  m_CstDecision.resize(m_ClassNb,0);
  m_LambdaQ.resize(m_ClassNb, MatrixType(m_FeatNb,m_FeatNb));

  RealType lambda;
  typedef itk::Statistics::CovarianceSampleFilter< itk::Statistics::Subsample< InputListSampleType > > CovarianceEstimatorType;
  typename CovarianceEstimatorType::Pointer covarianceEstimator = CovarianceEstimatorType::New();
  for ( unsigned int i = 0; i < m_ClassNb; ++i )
  {
    // Estimate covariance matrices, mean vectors and proportions
    m_NbSpl[i] = m_ClassSamples[i]->Size();
    m_Proportion[i] = (float) m_NbSpl[i] / (float) sampleNb;

    covarianceEstimator->SetInput( m_ClassSamples[i] );
    covarianceEstimator->Update();

    m_Covariances[i] = covarianceEstimator->GetCovarianceMatrix().GetVnlMatrix();
    m_Means[i] = VectorType(covarianceEstimator->GetMean().GetDataPointer(),m_FeatNb);

    // Decompose covariance matrix in eigenvalues/eigenvectors
    Decomposition(m_Covariances[i], m_Q[i], m_EigenValues[i]);

    // Precompute lambda^(-1/2) * Q and log(det lambda)
    for (int j = 0; j < m_FeatNb; ++j)
    {
      lambda = 1 / sqrt(m_EigenValues[i][j] + m_Tau);
      // Transposition and row multiplication at the same time
      m_LambdaQ[i].set_row(j,lambda*m_Q[i].get_column(j));

      m_CstDecision[i] += log(m_EigenValues[i][j] + m_Tau);
    }

    m_CstDecision[i] += -2*log(m_Proportion[i]);
  }
}

template <class TInputValue, class TOutputValue>
typename GMMMachineLearningModel<TInputValue,TOutputValue>
::TargetSampleType
GMMMachineLearningModel<TInputValue,TOutputValue>
::Predict(const InputSampleType & input, ConfidenceValueType *quality) const
{
  if (quality != NULL)
  {
    if (!this->HasConfidenceIndex())
    {
      itkExceptionMacro("Confidence index not available for this classifier !");
    }
  }

  VectorType input_c(m_FeatNb);
  // Compute decision function
  std::vector<RealType> decisionFct(m_CstDecision);
  VectorType lambdaQInputC;
  for (int i = 0; i < m_ClassNb; ++i)
  {
    vnl_copy(vnl_vector<InputValueType>(input.GetDataPointer(), m_FeatNb),input_c);
    input_c -= m_Means[i];
    lambdaQInputC = m_LambdaQ[i] * input_c;

    // Add sum of squared elements
    decisionFct[i] += lambdaQInputC.squared_magnitude();
  }

  int argmin = std::distance(decisionFct.begin(), std::min_element(decisionFct.begin(), decisionFct.end()));

  TargetSampleType res;
  res[0] = m_MapOfIndices.at(argmin);

  return res;
}

template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::Save(const std::string & filename, const std::string & name)
{
  // create and open a character archive for output
  std::ofstream ofs(filename.c_str(), std::ios::out);

  // Store single value data
  ofs << m_ClassNb << std::endl;
  ofs << m_FeatNb << std::endl;
  ofs << m_Tau << std::endl;

  // Store label mapping (only one way)
  typedef typename std::map<TargetValueType, int>::const_iterator MapIterator;
  for (MapIterator classMapIter = m_MapOfClasses.begin(); classMapIter != m_MapOfClasses.end(); classMapIter++)
    ofs << classMapIter->first << " " << classMapIter->second << " ";
  ofs << std::endl;

  // Store vector of nb of samples
  for (int i = 0; i < m_ClassNb; ++i)
    ofs << m_NbSpl[i] << " ";
  ofs << std::endl;

  // Set writing precision (need c++11 to not hardcode value of double precision)
  // ofs.precision(std::numeric_limits<double>::max_digits10);
  ofs.precision(17);

  // Store vector of proportion of samples
  for (int i = 0; i < m_ClassNb; ++i)
    ofs << m_Proportion[i] << " ";
  ofs << std::endl;

  // Store vector of mean vector (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
  {
    for (int j = 0; j < m_FeatNb; ++j)
      ofs << m_Means[i][j] << " ";

    ofs << std::endl;
  }

  // Store vector of covariance matrices (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
  {
    for (int j = 0; j < m_FeatNb; ++j)
    {
      for (int k = 0; k < m_FeatNb; ++k)
        ofs << m_Covariances[i](j,k) << " ";
    }
    ofs << std::endl;
  }

  // Store vector of eigenvalues vector (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
  {
    for (int j = 0; j < m_FeatNb; ++j)
      ofs << m_EigenValues[i][j] << " ";

    ofs << std::endl;
  }

  // Store vector of eigenvectors matrices (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
  {
    for (int j = 0; j < m_FeatNb; ++j)
    {
      for (int k = 0; k < m_FeatNb; ++k)
        ofs << m_Q[i](j,k) << " ";
    }
    ofs << std::endl;
  }

  // Store vector of eigenvalues^(-1/2) * Q.T matrices (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
  {
    for (int j = 0; j < m_FeatNb; ++j)
    {
      for (int k = 0; k < m_FeatNb; ++k)
        ofs << m_LambdaQ[i](j,k) << " ";
    }
    ofs << std::endl;
  }

  // Store vector of scalar (logdet cov - 2*log proportion)
  for (int i = 0; i < m_ClassNb; ++i)
    ofs << m_CstDecision[i] << " ";
  ofs << std::endl;
}

template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::Load(const std::string & filename, const std::string & name)
{
  std::ifstream ifs(filename.c_str(), std::ios::in);

  std::string test;
  // ifs >> test;

  // Load single value data
  ifs >> m_ClassNb;
  ifs >> m_FeatNb;
  ifs >> m_Tau;

  // Allocation
  m_NbSpl.resize(m_ClassNb);
  m_Proportion.resize(m_ClassNb);
  m_Means.resize(m_ClassNb,VectorType(m_FeatNb));
  m_Covariances.resize(m_ClassNb,MatrixType(m_FeatNb,m_FeatNb));
  m_EigenValues.resize(m_ClassNb,VectorType(m_FeatNb));
  m_Q.resize(m_ClassNb,MatrixType(m_FeatNb,m_FeatNb));
  m_LambdaQ.resize(m_ClassNb,MatrixType(m_FeatNb,m_FeatNb));
  m_CstDecision.resize(m_ClassNb);

  // Load label mapping (only one way)
  TargetValueType lab;
  int idLab;
  for (int i = 0; i < m_ClassNb; ++i)
  {
    ifs >> lab;
    ifs >> idLab;
    m_MapOfClasses[lab]   = idLab;
    m_MapOfIndices[idLab] = lab;
  }

  // Load vector of nb of samples
  for (int i = 0; i < m_ClassNb; ++i)
    ifs >> m_NbSpl[i];

  // Load vector of proportion of samples
  for (int i = 0; i < m_ClassNb; ++i)
    ifs >> m_Proportion[i];

  // Load vector of mean vector (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
    for (int j = 0; j < m_FeatNb; ++j)
      ifs >> m_Means[i][j];

  // Load vector of covariance matrices (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
    for (int j = 0; j < m_FeatNb; ++j)
      for (int k = 0; k < m_FeatNb; ++k)
        ifs >> m_Covariances[i](j,k);

  // Load vector of eigenvalues vector (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
    for (int j = 0; j < m_FeatNb; ++j)
      ifs >> m_EigenValues[i][j];

  // Load vector of eigenvectors matrices (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
    for (int j = 0; j < m_FeatNb; ++j)
      for (int k = 0; k < m_FeatNb; ++k)
        ifs >> m_Q[i](j,k);

  // Load vector of eigenvalues^(-1/2) * Q.T matrices (one by line)
  for (int i = 0; i < m_ClassNb; ++i)
    for (int j = 0; j < m_FeatNb; ++j)
      for (int k = 0; k < m_FeatNb; ++k)
        ifs >> m_LambdaQ[i](j,k);

  // Load vector of scalar (logdet cov - 2*log proportion)
  for (int i = 0; i < m_ClassNb; ++i)
    ifs >> m_CstDecision[i];
}

template <class TInputValue, class TOutputValue>
bool
GMMMachineLearningModel<TInputValue,TOutputValue>
::CanReadFile(const std::string & file)
{
  // std::ifstream ifs;
  // ifs.open(file.c_str());

  // if(!ifs)
  // {
  //   std::cerr<<"Could not read file "<<file<<std::endl;
  //   return false;
  // }

  // while (!ifs.eof())
  // {
  //   std::string line;
  //   std::getline(ifs, line);

  //   if (line.find(CV_TYPE_NAME_ML_NBAYES) != std::string::npos)
  //   {
  //      //std::cout<<"Reading a "<<CV_TYPE_NAME_ML_NBAYES<<" model"<<std::endl;
  //      return true;
  //   }
  // }
  // ifs.close();
  return false;
}

template <class TInputValue, class TOutputValue>
bool
GMMMachineLearningModel<TInputValue,TOutputValue>
::CanWriteFile(const std::string & itkNotUsed(file))
{
  return false;
}

template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  // Call superclass implementation
  Superclass::PrintSelf(os,indent);
}


template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::SetMapOfClasses(const std::map<TargetValueType, int>& mapOfClasses)
{
  m_MapOfClasses = mapOfClasses;
}

template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::SetMapOfIndices(const std::map<int, TargetValueType>& mapOfIndices)
{
 m_MapOfIndices = mapOfIndices;
}

template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::AddMean(VectorType vector)
{
  m_Means.push_back(vector);
}

template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::AddCovMatrix(MatrixType covMatrix)
{
  m_Covariances.push_back(covMatrix);
}

template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::AddNbSpl(unsigned long n)
{
  m_NbSpl.push_back(n);
}


} //end namespace otb

#endif