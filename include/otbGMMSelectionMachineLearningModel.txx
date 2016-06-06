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

template <class TInputValue, class TOutputValue>
GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::GMMSelectionMachineLearningModel()
{
}


template <class TInputValue, class TOutputValue>
GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::~GMMSelectionMachineLearningModel()
{
}

template <class TInputValue, class TOutputValue>
void
GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::AddInstanceToFold(std::vector<InstanceIdentifier> & input, int start, int end)
{
  int currentSize = m_fold.size()
  m_fold.resize(currentSize + end - start);
  for (int i = 0; i < end-start; ++i)
    m_fold[currentSize+i] = input[start+i];
}

template <class TInputValue, class TOutputValue>
void
GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::ForwardSelection(std::string method, std::string criterion, int selestedVarNb, int nfold)
{

  if ( criterion.compare("accuracy") == 0 || criterion.compare("kappa") == 0 || criterion.compare("F1mean") == 0)
  {
    std::vector<GMMSelectionMachineLearningModel<TInputValue, TTargetValue> > submodelCv(nfold);
    typename CovarianceEstimatorType::Pointer covarianceEstimator = CovarianceEstimatorType::New();
    MeanVectorType meanFold;
    MatrixType covarianceFold;

    for (unsigned int i = 0; i < m_classNb; ++i)
    {
      std::srand ( unsigned ( std::time(0) ) );
      std::vector<InstanceIdentifier> indices;
      for (unsigned j=0; j<m_NbSpl[i]; ++j)
        indices.push_back(m_classSamples[i]->GetInstanceIdentifier(j));

      std::random_shuffle ( indices.begin(), indices.end() );

      unsigned nbSplFold = m_NbSpl[i]/nfold; // to verify

      for (int j = 0; j < nfold; ++j)
      {
        // Add
        submodelCv[j].AddInstanceToFold(indices,j*m_NbSpl[i]/nfold,(j+1)*m_NbSpl[i]/nfold);

        // Update model for each fold
        submodelCv[j].SetMapOfClasses(m_MapOfClasses);
        submodelCv[j].SetMapOfIndices(m_MapOfIndices);
        submodelCv[j].SetClassNb(m_classNb);
        submodelCv[j].SetFeatNb(m_featNb);

        covarianceEstimator->SetInput( /* list sample input*/ );
        covarianceEstimator->Update();

        covarianceFold = covarianceEstimator->GetCovarianceMatrix();
        meanFold       = covarianceEstimator->GetMean();



        submodelCv[j].AddNbSpl(nbSplFold);
        submodelCv[j].AddMean( (1/(m_NbSpl[i] - nbSplFold)) * (m_NbSpl[i] * m_Means[i] - nbSplFold * meanFold) );
        submodelCv[j].AddCovMatrix( (1/(m_NbSpl[i]-nbSplFold-1)) * ( (m_NbSpl[i]-1)*m_Covariances[i] - (nbSplFold-1)*covarianceFold - m_NbSpl[i]*nbSplFold/(m_NbSpl[i]-nbSplFold) * (m_Means[i]-meanFold) * (m_Means[i]-meanFold) ) );
        submodelCv[j].UpdateProportion();
      }


// # Precompute cst
// model_pre_cv[k].logprop = 2*sp.log(model_pre_cv[k].prop)




    }

    for (int j = 0; j < nfold; ++j)
      submodelCv[j].UpdateProportion();
  }

}

template <class TInputValue, class TOutputValue>
void
GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::FloatingForwardSelection(std::string method, std::string criterion, int selestedVarNb, int nfold)
{

}

template <class TInputValue, class TOutputValue>
void
GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  // Call superclass implementation
  Superclass::PrintSelf(os,indent);
}

} //end namespace otb

#endif