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
::AddInstanceToFold(std::vector<InstanceIdentifier> & input, int start, int end)
{
  m_Fold.push_back( ClassSampleType::New() );
  for (int i = 0; i < end-start; ++i)
    (*m_Fold.end())->AddInstance( input[i] );
}

template <class TInputValue, class TOutputValue>
void
GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::UpdateProportion()
{
  unsigned totalNb = 0;
  for (int i = 0; i < this->m_ClassNb; ++i)
    totalNb += this->m_NbSpl[i];

  for (int i = 0; i < this->m_ClassNb; ++i)
  {
    this->m_Proportion[i] = (double) this->m_NbSpl[i] / (double) totalNb;
    m_Logprop[i]    = log(this->m_Proportion[i]);
  }

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ForwardSelection(std::string criterion, int selectedVarNb, int nfold)
{

  // Creation of submodel for cross-validation
  if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("F1mean") == 0))
  {
    // Allocation
    std::vector<GMMSelectionMachineLearningModel<TInputValue, TTargetValue>::Pointer > submodelCv(nfold);
    typedef itk::Statistics::CovarianceSampleFilter< itk::Statistics::Subsample< InputListSampleType > > CovarianceEstimatorType;
    typename CovarianceEstimatorType::Pointer covarianceEstimator = CovarianceEstimatorType::New();
    VectorType meanFold;
    MatrixType covarianceFold, adjustedMean;

    for (unsigned int i = 0; i < this->m_ClassNb; ++i)
    {
      // Shuffle id of samples
      std::srand( unsigned( std::time(0) ) );
      std::vector<InstanceIdentifier> indices;
      for (unsigned j=0; j<this->m_NbSpl[i]; ++j)
        indices.push_back((this->m_ClassSamples[i])->GetInstanceIdentifier(j));

      std::random_shuffle( indices.begin(), indices.end() );

      unsigned nbSplFold = this->m_NbSpl[i]/nfold; // to verify

      for (int j = 0; j < nfold; ++j)
      {
        // Add subpart of id to fold
        if (j==nfold-1)
          submodelCv[j]->AddInstanceToFold(indices,j*this->m_NbSpl[i]/nfold,this->m_NbSpl[i]+1);
        else
          submodelCv[j]->AddInstanceToFold(indices,j*this->m_NbSpl[i]/nfold,(j+1)*this->m_NbSpl[i]/nfold);

        // Update model for each fold
        submodelCv[j]->SetMapOfClasses(this->m_MapOfClasses);
        submodelCv[j]->SetMapOfIndices(this->m_MapOfIndices);
        submodelCv[j]->SetClassNb(this->m_ClassNb);
        submodelCv[j]->SetFeatNb(this->m_FeatNb);

        covarianceEstimator->SetInput( submodelCv[j]->GetClassSamples(i) );
        covarianceEstimator->Update();

        covarianceFold = covarianceEstimator->GetCovarianceMatrix().GetVnlMatrix();
        meanFold       = VectorType(covarianceEstimator->GetMean().GetDataPointer(),this->m_FeatNb);

        submodelCv[j]->AddNbSpl(nbSplFold);
        submodelCv[j]->AddMean( (1/((RealType) this->m_NbSpl[i] - (RealType) nbSplFold)) * ((RealType) this->m_NbSpl[i] * this->m_Means[i] - (RealType) nbSplFold * meanFold) );
        adjustedMean = MatrixType((this->m_Means[i]-meanFold).data_block(), this->m_FeatNb, 1);
        submodelCv[j]->AddCovMatrix( (1/((RealType)this->m_NbSpl[i]-(RealType)nbSplFold-1)) * ( ((RealType)this->m_NbSpl[i]-1)*this->m_Covariances[i] - ((RealType)nbSplFold-1)*covarianceFold - (RealType)this->m_NbSpl[i]*(RealType)nbSplFold/((RealType)this->m_NbSpl[i]-(RealType)nbSplFold) * adjustedMean * adjustedMean.transpose() ) );
      }
    }

    for (int i = 0; i < nfold; ++i)
      submodelCv[i]->UpdateProportion();
  }

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::FloatingForwardSelection(std::string criterion, int selectedVarNb, int nfold)
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
