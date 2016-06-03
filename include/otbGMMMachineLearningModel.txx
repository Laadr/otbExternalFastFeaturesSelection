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
#include <vnl/vnl_matrix.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <vnl/algo/vnl_generalized_eigensystem.h>

namespace otb
{

template <class TInputValue, class TOutputValue>
GMMMachineLearningModel<TInputValue,TOutputValue>
::GMMMachineLearningModel():
  m_classNb(0),
  m_featNb(0),
  m_tau(0)
{
  m_CovarianceEstimator = CovarianceEstimatorType::New();
}


template <class TInputValue, class TOutputValue>
GMMMachineLearningModel<TInputValue,TOutputValue>
::~GMMMachineLearningModel()
{
}

/** Set m_tau and update m_lambdaQ and m_cstDecision */
template <class TInputValue, class TOutputValue>
void GMMMachineLearningModel<TInputValue,TOutputValue>
::SetTau(MatrixValueType tau)
{
  m_tau = tau;

  // Precompute lambda^(-1/2) * Q and log(det lambda)
  if (! m_Q.empty())
  {
    MatrixValueType lambda;
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
::Decomposition(MatrixType &inputMatrix, MatrixType &outputMatrix, itk::VariableLengthVector<MatrixValueType> &eigenValues)
{
  vnl_vector<double> vectValP;
  vnl_symmetric_eigensystem_compute( inputMatrix.GetVnlMatrix(), outputMatrix.GetVnlMatrix(), vectValP );


  if (m_tau == 0)
  {
    for (int i = 0; i < eigenValues.GetSize(); ++i)
    {
      if (eigenValues[i] < std::numeric_limits<MatrixValueType>::epsilon())
        eigenValues[i] = std::numeric_limits<MatrixValueType>::epsilon();
      else
        eigenValues[i] = vectValP[i];
    }
  }else{
    for (int i = 0; i < eigenValues.GetSize(); ++i)
      eigenValues[i] = vectValP[i];
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
  typename TargetListSampleType::ConstIterator refIterator = labels->Begin();
  typename InputListSampleType::ConstIterator inputIterator = samples->Begin();

  // Get number of samples
  unsigned long sampleNb = samples->Size();

  // Get number of features
  m_featNb = samples->GetMeasurementVectorSize();

  // Get number of classes and map indice with label
  while (refIterator != labels->End())
  {
    TargetValueType currentLabel = refIterator.GetMeasurementVector()[0];
    if (m_MapOfClasses.find(currentLabel) == m_MapOfClasses.end())
    {
      m_MapOfClasses[currentLabel] = m_classNb;
      m_MapOfIndices[m_classNb] = currentLabel;
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
  refIterator = labels->Begin();
  inputIterator = samples->Begin();
  while (inputIterator != samples->End())
  {
    TargetValueType currentLabel = refIterator.GetMeasurementVector()[0];
    m_classSamples[m_MapOfClasses[currentLabel]]->AddInstance( inputIterator.GetInstanceIdentifier() );
    ++inputIterator;
    ++refIterator;
  }

  // Estimate covariance matrices, mean vectors and proportions
  m_NbSpl.resize(m_classNb);
  m_Proportion.resize(m_classNb);
  m_Covariances.resize(m_classNb,MatrixType(m_featNb,m_featNb));
  m_Means.resize(m_classNb,MeanVectorType(m_featNb));
  for ( unsigned int i = 0; i < m_classNb; ++i )
  {
    m_NbSpl[i] = m_classSamples[i]->Size();
    m_Proportion[i] = (float) m_NbSpl[i] / (float) sampleNb;

    m_CovarianceEstimator->SetInput( m_classSamples[i] );
    m_CovarianceEstimator->Update();

    m_Covariances[i] = m_CovarianceEstimator->GetCovarianceMatrix();
    m_Means[i] = m_CovarianceEstimator->GetMean();
  }

  // Decompose covariance matrix in eigenvalues/eigenvectors
  m_Q.resize(m_classNb,MatrixType(m_featNb,m_featNb));
  itk::VariableLengthVector<MatrixValueType> newVector;
  newVector.SetSize(m_featNb);
  m_eigenValues.resize(m_classNb,newVector);

  for (int i = 0; i < m_classNb; ++i)
  {
    // Make decomposition
    Decomposition(m_Covariances[i], m_Q[i], m_eigenValues[i]);
  }

  // Precompute lambda^(-1/2) * Q and log(det lambda)
  MatrixValueType lambda;
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

  for (int i = 0; i < m_classNb; ++i)
    std::cout << m_NbSpl[i] << std::endl;
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

  itk::Array<MatrixValueType> input_c;
  input_c.SetSize(input.GetSize());
  // Compute decision function
  std::vector<MatrixValueType> decisionFct(m_cstDecision);
  itk::Array<MatrixValueType> lambdaQInputC;
  for (int i = 0; i < m_classNb; ++i)
  {
    for (int j = 0; j < m_featNb; ++j)
      input_c[j]= input[j] - m_Means[i][j];

    lambdaQInputC = m_lambdaQ[i].GetVnlMatrix() * input_c;

    for (int j = 0; j < m_featNb; ++j)
      decisionFct[i] += lambdaQInputC[j]*lambdaQInputC[j];
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
  std::ofstream ofs(filename.c_str(), std::ios::binary);

}

template <class TInputValue, class TOutputValue>
void
GMMMachineLearningModel<TInputValue,TOutputValue>
::Load(const std::string & filename, const std::string & name)
{

  std::ifstream ifs(filename.c_str(), std::ios::binary);

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
::AddMean(MeanVectorType vector)
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