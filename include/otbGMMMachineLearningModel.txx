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
  m_classNb(0),
  m_featNb(0),
  m_tau(0)
{
}


template <class TInputValue, class TOutputValue>
GMMMachineLearningModel<TInputValue,TOutputValue>
::~GMMMachineLearningModel()
{
}

/** Set m_tau and update m_lambdaQ and m_cstDecision */
template <class TInputValue, class TOutputValue>
void GMMMachineLearningModel<TInputValue,TOutputValue>
::SetTau(RealType tau)
{
  m_tau = tau;

  // Precompute lambda^(-1/2) * Q and log(det lambda)
  if (!m_Q.empty())
  {
    RealType lambda;
    m_cstDecision.resize(m_classNb,0);
    m_lambdaQ.resize(m_classNb, MatrixType(m_featNb,m_featNb));

    for (int i = 0; i < m_classNb; ++i)
    {
      for (int j = 0; j < m_featNb; ++j)
      {
        lambda = 1 / sqrt(m_eigenValues[i][j] + m_tau);
        for (int k = 0; k < m_featNb; ++k)
          m_lambdaQ[i](j,k) = lambda * m_Q[i](k,j);

        m_cstDecision[i] += log(m_eigenValues[i][j] + m_tau);
      }

      m_cstDecision[i] += -2*log(m_Proportion[i]);
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
  m_featNb = samples->GetMeasurementVectorSize();

  // Get number of classes and map indice with label
  TargetValueType currentLabel;
  while (refIterator != labels->End())
  {
    currentLabel = refIterator.GetMeasurementVector()[0];
    if (m_MapOfClasses.find(currentLabel) == m_MapOfClasses.end())
    {
      m_MapOfClasses[currentLabel] = m_classNb;
      m_MapOfIndices[m_classNb]    = currentLabel;
      ++m_classNb;
    }
    ++refIterator;
  }

  // Create one subsample set for each class
  m_classSamples.reserve(m_classNb);
  for ( unsigned int i = 0; i < m_classNb; ++i )
  {
    m_classSamples.push_back( ClassSampleType::New() );
    m_classSamples[i]->SetSample( samples );
  }
  inputIterator = samples->Begin();
  refIterator   = labels->Begin();
  while (inputIterator != samples->End())
  {
    currentLabel = refIterator.GetMeasurementVector()[0];
    m_classSamples[m_MapOfClasses[currentLabel]]->AddInstance( inputIterator.GetInstanceIdentifier() );
    ++inputIterator;
    ++refIterator;
  }

  // Estimate covariance matrices, mean vectors and proportions
  typedef itk::Statistics::CovarianceSampleFilter< itk::Statistics::Subsample< InputListSampleType > > CovarianceEstimatorType;
  typename CovarianceEstimatorType::Pointer covarianceEstimator = CovarianceEstimatorType::New();

  m_NbSpl.resize(m_classNb);
  m_Proportion.resize(m_classNb);
  m_Covariances.resize(m_classNb,MatrixType(m_featNb,m_featNb));
  m_Means.resize(m_classNb,VectorType(m_featNb));
  for ( unsigned int i = 0; i < m_classNb; ++i )
  {
    m_NbSpl[i] = m_classSamples[i]->Size();
    m_Proportion[i] = (float) m_NbSpl[i] / (float) sampleNb;

    covarianceEstimator->SetInput( m_classSamples[i] );
    covarianceEstimator->Update();

    m_Covariances[i] = covarianceEstimator->GetCovarianceMatrix().GetVnlMatrix();
    m_Means[i] = VectorType(covarianceEstimator->GetMean().GetDataPointer(),m_featNb);
  }

  // Decompose covariance matrix in eigenvalues/eigenvectors
  m_Q.resize(m_classNb,MatrixType(m_featNb,m_featNb));
  m_eigenValues.resize(m_classNb,VectorType(m_featNb));

  for (int i = 0; i < m_classNb; ++i)
  {
    // Make decomposition
    Decomposition(m_Covariances[i], m_Q[i], m_eigenValues[i]);
  }

  // Precompute lambda^(-1/2) * Q and log(det lambda)
  RealType lambda;
  m_cstDecision.resize(m_classNb,0);
  m_lambdaQ.resize(m_classNb, MatrixType(m_featNb,m_featNb));
  for (int i = 0; i < m_classNb; ++i)
  {
    for (int j = 0; j < m_featNb; ++j)
    {
      lambda = 1 / sqrt(m_eigenValues[i][j] + m_tau);
      // for (int k = 0; k < m_featNb; ++k)
      //   m_lambdaQ[i](j,k) = lambda * m_Q[i](k,j);
      // Transposition and row multiplication at the same time
      m_lambdaQ[i].set_row(j,lambda*m_Q[i].get_column(j));

      m_cstDecision[i] += log(m_eigenValues[i][j] + m_tau);
    }

    m_cstDecision[i] += -2*log(m_Proportion[i]);
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

  VectorType input_c(m_featNb);
  // Compute decision function
  std::vector<RealType> decisionFct(m_cstDecision);
  VectorType lambdaQInputC;
  for (int i = 0; i < m_classNb; ++i)
  {
    vnl_copy(vnl_vector<InputValueType>(input.GetDataPointer(), m_featNb),input_c);
    input_c -= m_Means[i];
    lambdaQInputC = m_lambdaQ[i] * input_c;

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
  ofs << m_classNb << std::endl;
  ofs << m_featNb << std::endl;
  ofs << m_tau << std::endl;

  // Store label mapping (only one way)
  typedef typename std::map<TargetValueType, int>::const_iterator MapIterator;
  for (MapIterator classMapIter = m_MapOfClasses.begin(); classMapIter != m_MapOfClasses.end(); classMapIter++)
    ofs << classMapIter->first << " " << classMapIter->second << " ";
  ofs << std::endl;

  // Store vector of nb of samples
  for (int i = 0; i < m_classNb; ++i)
    ofs << m_NbSpl[i] << " ";
  ofs << std::endl;

  // Set writing precision (need c++11 to not hardcode value of double precision)
  // ofs.precision(std::numeric_limits<double>::max_digits10);
  ofs.precision(17);

  // Store vector of proportion of samples
  for (int i = 0; i < m_classNb; ++i)
    ofs << m_Proportion[i] << " ";
  ofs << std::endl;

  // Store vector of mean vector (one by line)
  for (int i = 0; i < m_classNb; ++i)
  {
    for (int j = 0; j < m_featNb; ++j)
      ofs << m_Means[i][j] << " ";

    ofs << std::endl;
  }

  // Store vector of covariance matrices (one by line)
  for (int i = 0; i < m_classNb; ++i)
  {
    for (int j = 0; j < m_featNb; ++j)
    {
      for (int k = 0; k < m_featNb; ++k)
        ofs << m_Covariances[i](j,k) << " ";
    }
    ofs << std::endl;
  }

  // Store vector of eigenvalues vector (one by line)
  for (int i = 0; i < m_classNb; ++i)
  {
    for (int j = 0; j < m_featNb; ++j)
      ofs << m_eigenValues[i][j] << " ";

    ofs << std::endl;
  }

  // Store vector of eigenvectors matrices (one by line)
  for (int i = 0; i < m_classNb; ++i)
  {
    for (int j = 0; j < m_featNb; ++j)
    {
      for (int k = 0; k < m_featNb; ++k)
        ofs << m_Q[i](j,k) << " ";
    }
    ofs << std::endl;
  }

  // Store vector of eigenvalues^(-1/2) * Q.T matrices (one by line)
  for (int i = 0; i < m_classNb; ++i)
  {
    for (int j = 0; j < m_featNb; ++j)
    {
      for (int k = 0; k < m_featNb; ++k)
        ofs << m_lambdaQ[i](j,k) << " ";
    }
    ofs << std::endl;
  }

  // Store vector of scalar (logdet cov - 2*log proportion)
  for (int i = 0; i < m_classNb; ++i)
    ofs << m_cstDecision[i] << " ";
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
  ifs >> m_classNb;
  ifs >> m_featNb;
  ifs >> m_tau;

  // Allocation
  m_NbSpl.resize(m_classNb);
  m_Proportion.resize(m_classNb);
  m_Means.resize(m_classNb,VectorType(m_featNb));
  m_Covariances.resize(m_classNb,MatrixType(m_featNb,m_featNb));
  m_eigenValues.resize(m_classNb,VectorType(m_featNb));
  m_Q.resize(m_classNb,MatrixType(m_featNb,m_featNb));
  m_lambdaQ.resize(m_classNb,MatrixType(m_featNb,m_featNb));
  m_cstDecision.resize(m_classNb);

  // Load label mapping (only one way)
  TargetValueType lab;
  int idLab;
  for (int i = 0; i < m_classNb; ++i)
  {
    ifs >> lab;
    ifs >> idLab;
    m_MapOfClasses[lab]   = idLab;
    m_MapOfIndices[idLab] = lab;
  }

  // Load vector of nb of samples
  for (int i = 0; i < m_classNb; ++i)
    ifs >> m_NbSpl[i];

  // Load vector of proportion of samples
  for (int i = 0; i < m_classNb; ++i)
    ifs >> m_Proportion[i];

  // Load vector of mean vector (one by line)
  for (int i = 0; i < m_classNb; ++i)
    for (int j = 0; j < m_featNb; ++j)
      ifs >> m_Means[i][j];

  // Load vector of covariance matrices (one by line)
  for (int i = 0; i < m_classNb; ++i)
    for (int j = 0; j < m_featNb; ++j)
      for (int k = 0; k < m_featNb; ++k)
        ifs >> m_Covariances[i](j,k);

  // Load vector of eigenvalues vector (one by line)
  for (int i = 0; i < m_classNb; ++i)
    for (int j = 0; j < m_featNb; ++j)
      ifs >> m_eigenValues[i][j];

  // Load vector of eigenvectors matrices (one by line)
  for (int i = 0; i < m_classNb; ++i)
    for (int j = 0; j < m_featNb; ++j)
      for (int k = 0; k < m_featNb; ++k)
        ifs >> m_Q[i](j,k);

  // Load vector of eigenvalues^(-1/2) * Q.T matrices (one by line)
  for (int i = 0; i < m_classNb; ++i)
    for (int j = 0; j < m_featNb; ++j)
      for (int k = 0; k < m_featNb; ++k)
        ifs >> m_lambdaQ[i](j,k);

  // Load vector of scalar (logdet cov - 2*log proportion)
  for (int i = 0; i < m_classNb; ++i)
    ifs >> m_cstDecision[i];
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

template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::UpdateProportion()
{
  unsigned totalNb = 0;
  for (int i = 0; i < m_classNb; ++i)
    totalNb += m_NbSpl[i];

  for (int i = 0; i < m_classNb; ++i)
  {
    m_Proportion[i] = (float) m_NbSpl[i] / (float) totalNb;
  }

}


} //end namespace otb

#endif