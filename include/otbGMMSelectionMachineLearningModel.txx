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
  for (int i = 0; i < this->m_ClassNb; ++i)
    totalNb += this->m_NbSpl[i];

  this->m_Proportion.resize(this->m_ClassNb);
  m_Logprop.resize(this->m_ClassNb);
  for (int i = 0; i < this->m_ClassNb; ++i)
  {
    this->m_Proportion[i] = (double) this->m_NbSpl[i] / (double) totalNb;
    m_Logprop[i]          = (RealType) log(this->m_Proportion[i]);
  }

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Selection(std::string direction, std::string criterion, int selectedVarNb, int nfold)
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
          m_SubmodelCv[j]->AddInstanceToFold(this->GetInputListSample(), indices,j*nbSplFold,this->m_NbSpl[i]+1);
        else
          m_SubmodelCv[j]->AddInstanceToFold(this->GetInputListSample(), indices,j*nbSplFold,(j+1)*nbSplFold);

        // Update model for each fold
        m_SubmodelCv[j]->SetMapOfClasses(this->m_MapOfClasses);
        m_SubmodelCv[j]->SetMapOfIndices(this->m_MapOfIndices);
        m_SubmodelCv[j]->SetClassNb(this->m_ClassNb);
        m_SubmodelCv[j]->SetFeatNb(this->m_FeatNb);

        covarianceEstimator->SetInput( m_SubmodelCv[j]->GetClassSamples(i) );
        covarianceEstimator->Update();

        covarianceFold = covarianceEstimator->GetCovarianceMatrix().GetVnlMatrix();
        meanFold       = VectorType(covarianceEstimator->GetMean().GetDataPointer(),this->m_FeatNb);

        m_SubmodelCv[j]->AddNbSpl(nbSplFold);
        m_SubmodelCv[j]->AddMean( (1/((RealType) this->m_NbSpl[i] - (RealType) nbSplFold)) * ((RealType) this->m_NbSpl[i] * this->m_Means[i] - (RealType) nbSplFold * meanFold) );
        adjustedMean = MatrixType((this->m_Means[i]-meanFold).data_block(), this->m_FeatNb, 1);
        m_SubmodelCv[j]->AddCovMatrix( (1/((RealType)this->m_NbSpl[i]-(RealType)nbSplFold-1)) * ( ((RealType)this->m_NbSpl[i]-1)*this->m_Covariances[i] - ((RealType)nbSplFold-1)*covarianceFold - (RealType)this->m_NbSpl[i]*(RealType)nbSplFold/((RealType)this->m_NbSpl[i]-(RealType)nbSplFold) * adjustedMean * adjustedMean.transpose() ) ); // convert all unsigned in realType - ok?
      }
    }

    for (int i = 0; i < nfold; ++i)
      m_SubmodelCv[i]->UpdateProportion();
  }

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ForwardSelection(std::string criterion, int selectedVarNb)
{

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::FloatingForwardSelection(std::string criterion, int selectedVarNb)
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
