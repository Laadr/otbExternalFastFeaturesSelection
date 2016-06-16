#ifndef __otbGMMSelectionMachineLearningModel_txx
#define __otbGMMSelectionMachineLearningModel_txx

#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include "itkMacro.h"
#include "itkSubsample.h"
#include "itkSymmetricEigenAnalysis.h"
#include "otbGMMMachineLearningModel.h"

namespace otb
{

template <class TInputValue, class TTargetValue>
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::GMMSelectionMachineLearningModel()
{
}


template <class TInputValue, class TTargetValue>
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::~GMMSelectionMachineLearningModel()
{
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::AddInstanceToFold(typename InputListSampleType::Pointer samples, std::vector<InstanceIdentifier> & input, int start, int end)
{
  m_Fold.push_back( ClassSampleType::New() );
  m_Fold[m_Fold.size()-1]->SetSample( samples );
  for (int i = 0; i < end-start; ++i)
    m_Fold[m_Fold.size()-1]->AddInstance( input[i] );
}

template <class TInputValue, class TOutputValue>
void
GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::UpdateProportion()
{
  unsigned totalNb = 0;
  for (int i = 0; i < Superclass::m_ClassNb; ++i)
    totalNb += Superclass::m_NbSpl[i];

  Superclass::m_Proportion.resize(Superclass::m_ClassNb);
  m_Logprop.resize(Superclass::m_ClassNb);
  for (int i = 0; i < Superclass::m_ClassNb; ++i)
  {
    Superclass::m_Proportion[i] = (double) Superclass::m_NbSpl[i] / (double) totalNb;
    m_Logprop[i]          = (RealType) log(Superclass::m_Proportion[i]);
  }

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Selection(std::vector<int> & selectedVar, std::string direction, std::string criterion, int selectedVarNb, int nfold)
{

  // Creation of submodel for cross-validation
  if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("F1mean") == 0))
  {
    // Allocation
    for (int j = 0; j < nfold; ++j)
      m_SubmodelCv.push_back(GMMSelectionMachineLearningModel<TInputValue, TTargetValue>::New());
    typedef itk::Statistics::CovarianceSampleFilter< itk::Statistics::Subsample< InputListSampleType > > CovarianceEstimatorType;
    typename CovarianceEstimatorType::Pointer covarianceEstimator = CovarianceEstimatorType::New();
    VectorType meanFold;
    MatrixType covarianceFold, adjustedMean;

    for (unsigned int i = 0; i < Superclass::m_ClassNb; ++i)
    {
      // Shuffle id of samples
      std::srand( unsigned( std::time(0) ) );
      std::vector<InstanceIdentifier> indices;
      for (unsigned j=0; j<Superclass::m_NbSpl[i]; ++j)
        indices.push_back((Superclass::m_ClassSamples[i])->GetInstanceIdentifier(j));

      std::random_shuffle( indices.begin(), indices.end() );

      unsigned nbSplFold = Superclass::m_NbSpl[i]/nfold; // to verify

      for (int j = 0; j < nfold; ++j)
      {
        // Add subpart of id to fold
        if (j==nfold-1)
          m_SubmodelCv[j]->AddInstanceToFold(Superclass::GetInputListSample(), indices,j*nbSplFold,Superclass::m_NbSpl[i]+1);
        else
          m_SubmodelCv[j]->AddInstanceToFold(Superclass::GetInputListSample(), indices,j*nbSplFold,(j+1)*nbSplFold);

        // Update model for each fold
        m_SubmodelCv[j]->SetMapOfClasses(Superclass::m_MapOfClasses);
        m_SubmodelCv[j]->SetMapOfIndices(Superclass::m_MapOfIndices);
        m_SubmodelCv[j]->SetClassNb(Superclass::m_ClassNb);
        m_SubmodelCv[j]->SetFeatNb(Superclass::m_FeatNb);

        covarianceEstimator->SetInput( m_SubmodelCv[j]->GetClassSamples(i) );
        covarianceEstimator->Update();

        covarianceFold = covarianceEstimator->GetCovarianceMatrix().GetVnlMatrix();
        meanFold       = VectorType(covarianceEstimator->GetMean().GetDataPointer(),Superclass::m_FeatNb);

        m_SubmodelCv[j]->AddNbSpl(nbSplFold);
        m_SubmodelCv[j]->AddMean( (1/((RealType) Superclass::m_NbSpl[i] - (RealType) nbSplFold)) * ((RealType) Superclass::m_NbSpl[i] * Superclass::m_Means[i] - (RealType) nbSplFold * meanFold) );
        adjustedMean = MatrixType((Superclass::m_Means[i]-meanFold).data_block(), Superclass::m_FeatNb, 1);
        m_SubmodelCv[j]->AddCovMatrix( (1/((RealType)Superclass::m_NbSpl[i]-(RealType)nbSplFold-1)) * ( ((RealType)Superclass::m_NbSpl[i]-1)*Superclass::m_Covariances[i] - ((RealType)nbSplFold-1)*covarianceFold - (RealType)Superclass::m_NbSpl[i]*(RealType)nbSplFold/((RealType)Superclass::m_NbSpl[i]-(RealType)nbSplFold) * adjustedMean * adjustedMean.transpose() ) ); // convert all unsigned in realType - ok?
      }
    }

    for (int i = 0; i < nfold; ++i)
      m_SubmodelCv[i]->UpdateProportion();
  }

  if (direction.compare("forward") == 0)
    ForwardSelection(selectedVar, criterion, selectedVarNb);
  else if (direction.compare("sffs") == 0)
    FloatingForwardSelection(selectedVar, criterion, selectedVarNb);
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ForwardSelection(std::vector<int> & selectedVar, std::string criterion, int selectedVarNb)
{
  // Initialization
  int currentSelectedVarNb = 0;
  RealType maxValue;
  std::vector<RealType> criterionBestValues;
  std::list<int> variablesPool;
  variablesPool.resize(Superclass::m_FeatNb);
  for (int i = 0; i < Superclass::m_FeatNb; ++i)
  for (std::list<int>::iterator it = variablesPool.begin(); it != variablesPool.end(); it++)
    *it = i;

  // Start the forward search
  while ((currentSelectedVarNb<selectedVarNb)&&(!variablesPool.empty()))
  {

    std::vector<RealType> criterionVal(variablesPool.size(),0);

    // Compute criterion function
    if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("F1mean") == 0) )
    {
      ComputeClassifRate(criterionVal,"forward",variablesPool,selectedVar,criterion);
    }
    else if (criterion.compare("JM") == 0)
    {
      ComputeJM(criterionVal,"forward",variablesPool,selectedVar);
    }
    else if (criterion.compare("divKL") == 0)
    {
      ComputeDivKL(criterionVal,"forward",variablesPool,selectedVar);
    }

    // Select the variable that provides the highest criterion value
    maxValue = *(std::max_element(criterionVal.begin(), criterionVal.end()));
    criterionBestValues.push_back(maxValue);

    // Add it to selected var and delete it from the pool
    selectedVar.push_back(maxValue);
    variablesPool.remove(maxValue);

    currentSelectedVarNb++;
  }

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::FloatingForwardSelection(std::vector<int> & selectedVar, std::string criterion, int selectedVarNb)
{

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ComputeClassifRate(std::vector<RealType> & criterionVal, const std::string direction, const std::list<int> & variablesPool, const std::vector<int> & selectedVar, const std::string criterion)
{

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ComputeJM(std::vector<RealType> & criterionVal, const std::string direction, const std::list<int> & variablesPool, const std::vector<int> & selectedVar)
{

  // Compute all possible update of 0.5* log det cov(idx)
  std::vector<std::vector<RealType> > halfedLogdet(Superclass::m_ClassNb, std::vector<RealType>(variablesPool.size()));
  if (selectedVar.empty())
  {
    for (int c = 0; c < Superclass::m_ClassNb; ++c)
      for (int j = 0; j < variablesPool.size(); ++j)
        halfedLogdet[c][j] = 0.5*log(Superclass::m_Covariances[c](j,j));
  }
  else
  {
    MatrixType Q(Superclass::m_FeatNb,Superclass::m_FeatNb);
    VectorType eigenValues(Superclass::m_FeatNb);
    for (int c = 0; c < Superclass::m_ClassNb; ++c)
    {
      // Superclass::Decomposition(m_Covariances, Q, eigenValues);
    }
  }


// halfedLogdet  = sp.zeros((model.C,variables.size))

// if len(idx)==0:
//     for c in xrange(model.C):
//         for k,var in enumerate(variables):
//             halfedLogdet[c,k] = 0.5*sp.log(model.cov[c,var,var])
// else:
//     for c in xrange(model.C):
//         vp,Q,_ = model.decomposition(model.cov[c,idx,:][:,idx])
//         logdet = sp.sum(sp.log(vp))
//         invCov = sp.dot(Q,((1/vp)*Q).T)
//         for k,var in enumerate(variables):
//             if direction=='forward':
//                 alpha = model.cov[c,var,var] - sp.dot(model.cov[c,var,:][idx], sp.dot(invCov,model.cov[c,var,:][idx].T) )
//             elif direction=='backward':
//                 alpha = 1/invCov[k,k]

//             if alpha < eps:
//                 alpha = eps
//             halfedLogdet[c,k]  = 0.5*( sp.log(alpha) + logdet)
//     del vp,Q,alpha,invCov

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ComputeDivKL(std::vector<RealType> & criterionVal, const std::string direction, const std::list<int> & variablesPool, const std::vector<int> & selectedVar)
{

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  // Call superclass implementation
  Superclass::PrintSelf(os,indent);
}

template <class TInputValue, class TTargetValue>
typename GMMSelectionMachineLearningModel<TInputValue,TTargetValue>::ClassSamplePointer
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::GetClassSamples(int classId)
{
  return m_Fold[classId];
}


} //end namespace otb

#endif
