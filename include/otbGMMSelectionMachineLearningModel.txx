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
::ExtractVectorToColMatrix(const std::vector<int> & indexes, const VectorType& input, MatrixType& ouput)
{
  for (int i = 0; i < indexes.size(); ++i)
    ouput(i,0) = input[indexes[i]];
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ExtractSubSymmetricMatrix(const std::vector<int> & indexes, const MatrixType& input, MatrixType& ouput)
{
  for (int i = 0; i < indexes.size(); ++i)
  {
    ouput(i,i) = input(indexes[i],indexes[i]);
    for (int j = i+1; j < indexes.size(); ++j)
    {
      ouput(i,j) = input(indexes[i],indexes[j]);
      ouput(j,i) = ouput(i,j);
    }
  }
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ExtractReducedColumn(const int colIndex, const std::vector<int> & indexesRow, const MatrixType& input, MatrixType& ouput)
{
  for (int i = 0; i < indexesRow.size(); ++i)
    ouput(i,0) = input(indexesRow[i],colIndex);
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
    m_Logprop[i]                = (RealType) log(Superclass::m_Proportion[i]);
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
    ForwardSelection(criterion, selectedVarNb);
  else if (direction.compare("sffs") == 0)
    FloatingForwardSelection(criterion, selectedVarNb);
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ForwardSelection(std::string criterion, int selectedVarNb)
{
  // Initialization
  int currentSelectedVarNb = 0;
  RealType argMaxValue;
  std::vector<RealType> criterionBestValues;
  std::vector<int> variablesPool;
  variablesPool.resize(Superclass::m_FeatNb);
  m_SelectedVar.clear();
  for (int i = 0; i < Superclass::m_FeatNb; ++i)
    variablesPool[i] = i;

  // Start the forward search
  while ((currentSelectedVarNb<selectedVarNb)&&(!variablesPool.empty()))
  {

    std::vector<RealType> criterionVal(variablesPool.size(),0);

    // Compute criterion function
    if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("F1mean") == 0) )
    {
      ComputeClassifRate(criterionVal,"forward",variablesPool,criterion);
    }
    else if (criterion.compare("JM") == 0)
    {
      ComputeJM(criterionVal,"forward",variablesPool);
    }
    else if (criterion.compare("divKL") == 0)
    {
      ComputeDivKL(criterionVal,"forward",variablesPool);
    }

    // Select the variable that provides the highest criterion value
    argMaxValue = std::distance(criterionVal.begin(), std::max_element(criterionVal.begin(), criterionVal.end()));
    criterionBestValues.push_back(criterionVal[argMaxValue]);

    for (typename std::vector<RealType>::iterator it = criterionBestValues.begin(); it != criterionBestValues.end(); ++it)
    {
      std::cout << ' ' << *it;
    }
    std::cout << std::endl;

    // Add it to selected var and delete it from the pool
    m_SelectedVar.push_back(variablesPool[argMaxValue]);
    variablesPool.erase(variablesPool.begin()+argMaxValue);

    currentSelectedVarNb++;
  }
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
::ComputeClassifRate(std::vector<RealType> & criterionVal, const std::string direction, std::vector<int> & variablesPool, const std::string criterion)
{

}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ComputeJM(std::vector<RealType> & JM, const std::string direction, std::vector<int> & variablesPool)
{

  int selectedVarNb = m_SelectedVar.size();

  // Compute all possible update of 0.5* log det cov(idx)
  std::vector<std::vector<RealType> > halfedLogdet(Superclass::m_ClassNb, std::vector<RealType>(variablesPool.size()));
  if (m_SelectedVar.empty())
  {
    for (int c = 0; c < Superclass::m_ClassNb; ++c)
      for (int j = 0; j < variablesPool.size(); ++j)
        halfedLogdet[c][j] = 0.5*log(Superclass::m_Covariances[c](j,j));

    RealType md, cs, bij;

    for (int c1 = 0; c1 < Superclass::m_ClassNb; ++c1)
    {
      for (int c2 = c1+1; c2 < Superclass::m_ClassNb; ++c2)
      {
        std::vector<int>::iterator varIt = variablesPool.begin();
        for (int j = 0; j < variablesPool.size(); ++j)
        {
          md = Superclass::m_Means[c1][*varIt] - Superclass::m_Means[c2][*varIt];
          cs = Superclass::m_Covariances[c1](*varIt,*varIt) + Superclass::m_Covariances[c2](*varIt,*varIt);

          bij   = md*(0.25/cs)*md + 0.5*(log(cs) - halfedLogdet[c1][j] - halfedLogdet[c2][j]); // NB: md*(0.25/cs)*md = md*(2/cs)*md.T 8
          JM[j] += Superclass::m_Proportion[c1] * Superclass::m_Proportion[c2] * sqrt(2*(1-exp(-bij)));

          varIt++;
        }
      }
    }
  }
  else
  {
    std::vector<MatrixType> subCovariances(Superclass::m_ClassNb,MatrixType(selectedVarNb,selectedVarNb));
    MatrixType Q(selectedVarNb,selectedVarNb);
    MatrixType invCov(selectedVarNb,selectedVarNb);
    VectorType eigenValues(selectedVarNb);
    RealType logdet, alpha;
    MatrixType u(selectedVarNb,1);

    for (int c = 0; c < Superclass::m_ClassNb; ++c)
    {
      ExtractSubSymmetricMatrix(m_SelectedVar,Superclass::m_Covariances[c],subCovariances[c]);
      Superclass::Decomposition(subCovariances[c], Q, eigenValues);

      invCov = Q * (vnl_diag_matrix<RealType>(eigenValues).invert_in_place() * Q.transpose());
      logdet = eigenValues.apply(log).sum();

      std::vector<int>::iterator varIt = variablesPool.begin();
      for (int j = 0; j < variablesPool.size(); ++j)
      {
        if (direction.compare("forward")==0)
        {
          ExtractReducedColumn(*varIt,m_SelectedVar,Superclass::m_Covariances[c],u);
          alpha = Superclass::m_Covariances[c](*varIt,*varIt) - (u.transpose() * (invCov * u))(0,0);
          varIt++;
        }
        else if (direction.compare("backward")==0)
        {
          alpha = invCov(j,j); // actually corresponds to 1/alpha from report
        }

        if (alpha < std::numeric_limits<RealType>::epsilon())
          alpha = std::numeric_limits<RealType>::epsilon();
        halfedLogdet[c][j] = 0.5* (log(alpha) + logdet);
      }
    }

    MatrixType cs(selectedVarNb,selectedVarNb);
    RealType logdet_c1c2, cst_feat, bij;
    MatrixType md(selectedVarNb,1);
    MatrixType extractUTmp(selectedVarNb,1);

    // Extract means
    std::vector<MatrixType> subMeans(Superclass::m_ClassNb,MatrixType(selectedVarNb,1));
    for (int c = 0; c < Superclass::m_ClassNb; ++c)
      ExtractVectorToColMatrix(m_SelectedVar, Superclass::m_Means[c], subMeans[c]);

    // Compute JM
    for (int c1 = 0; c1 < Superclass::m_ClassNb; ++c1)
    {
      for (int c2 = c1+1; c2 < Superclass::m_ClassNb; ++c2)
      {
        cs = 0.5*(subCovariances[c1] + subCovariances[c2]);
        Superclass::Decomposition(cs, Q, eigenValues);

        invCov = Q * (vnl_diag_matrix<RealType>(eigenValues).invert_in_place() * Q.transpose());
        logdet = eigenValues.apply(log).sum();

        std::vector<int>::iterator varIt = variablesPool.begin();
        for (int j = 0; j < variablesPool.size(); ++j)
        {
          if (direction.compare("forward")==0)
          {
            md = subMeans[c1].extract(selectedVarNb,1) - subMeans[c2].extract(selectedVarNb,1);

            ExtractReducedColumn(*varIt,m_SelectedVar,Superclass::m_Covariances[c1],u);
            ExtractReducedColumn(*varIt,m_SelectedVar,Superclass::m_Covariances[c2],extractUTmp);
            u = 0.5*(u+extractUTmp);

            alpha = 0.5*(Superclass::m_Covariances[c1](*varIt,*varIt) + Superclass::m_Covariances[c2](*varIt,*varIt)) - (u.transpose() * (invCov * u))(0,0);
            if (alpha < std::numeric_limits<RealType>::epsilon())
              alpha = std::numeric_limits<RealType>::epsilon();

            logdet_c1c2 = logdet + log(alpha) + (selectedVarNb+1)*log(2);

            cst_feat = alpha * pow( ( ((-1/alpha)*(u.transpose()*invCov)*md)(0,0) + (Superclass::m_Means[c1][*varIt] - Superclass::m_Means[c2][*varIt])/alpha), 2);

            varIt++;
          }
          else if (direction.compare("backward")==0)
          {
            alpha = 1/invCov(j,j);
            if (alpha < std::numeric_limits<RealType>::epsilon())
              alpha = std::numeric_limits<RealType>::epsilon();

            logdet_c1c2 = logdet - log(alpha) + (selectedVarNb-1)*log(2);

            ExtractReducedColumn(j,m_SelectedVar,invCov,u);
            cst_feat = - alpha * pow( (u.transpose()*md)(0,0), 2);
          }

          bij = (1/8.) * (md.transpose() * (invCov*md))(0,0) + 0.5*(logdet_c1c2 - halfedLogdet[c1][j] - halfedLogdet[c2][j]);
          JM[j] += Superclass::m_Proportion[c1] * Superclass::m_Proportion[c2] * sqrt(2*(1-exp(-bij)));
        }
      }
    }
  }
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ComputeDivKL(std::vector<RealType> & criterionVal, const std::string direction, std::vector<int> & variablesPool)
{

}

template <class TInputValue, class TOutputValue>
typename GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::TargetSampleType
GMMSelectionMachineLearningModel<TInputValue,TOutputValue>
::Predict(const InputSampleType & input, ConfidenceValueType *quality) const
{
  TargetSampleType res;
  // res[0] = m_MapOfIndices.at(argmin);

  return res;
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

template <class TInputValue, class TTargetValue>
std::vector<int>
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::GetSelectedVar()
{
  return m_SelectedVar;
}

} //end namespace otb

#endif
