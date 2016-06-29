#ifndef __otbGMMSelectionMachineLearningModel_txx
#define __otbGMMSelectionMachineLearningModel_txx

#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include <vector>
#include <algorithm>

#include "otbConfusionMatrixCalculator.h"
#include "vnl/vnl_trace.h"

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
::ExtractVector(const std::vector<int> & indexes, const VectorType& input, VectorType& ouput)
{
  for (int i = 0; i < indexes.size(); ++i)
    ouput[i] = input[indexes[i]];
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
  for (int i = start; i < end; ++i)
    m_Fold[m_Fold.size()-1]->AddInstance( input[i] );
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
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
    m_Logprop[i]                = 2* (RealType) log(Superclass::m_Proportion[i]);
  }
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Selection(std::string direction, std::string criterion, int selectedVarNb, int nfold, int seed)
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
      std::srand( unsigned( seed ) );
      std::srand( unsigned( 0 ) );
      std::vector<InstanceIdentifier> indices;
      for (unsigned j=0; j<Superclass::m_NbSpl[i]; ++j)
        indices.push_back((Superclass::m_ClassSamples[i])->GetInstanceIdentifier(j));

      std::random_shuffle( indices.begin(), indices.end() );

      unsigned nbSplFold = Superclass::m_NbSpl[i]/nfold; // to verify

      for (int j = 0; j < nfold; ++j)
      {
        // Add subpart of id to fold
        if (j==nfold-1)
        {
          m_SubmodelCv[j]->AddInstanceToFold(Superclass::GetInputListSample(), indices,j*nbSplFold,Superclass::m_NbSpl[i]);
          m_SubmodelCv[j]->AddNbSpl(Superclass::m_NbSpl[i] - j*nbSplFold);
        }
        else
        {
          m_SubmodelCv[j]->AddInstanceToFold(Superclass::GetInputListSample(), indices,j*nbSplFold,(j+1)*nbSplFold);
          m_SubmodelCv[j]->AddNbSpl(nbSplFold);
        }

        // Update model for each fold
        m_SubmodelCv[j]->SetMapOfClasses(Superclass::m_MapOfClasses);
        m_SubmodelCv[j]->SetMapOfIndices(Superclass::m_MapOfIndices);
        m_SubmodelCv[j]->SetClassNb(Superclass::m_ClassNb);
        m_SubmodelCv[j]->SetFeatNb(Superclass::m_FeatNb);


        covarianceEstimator->SetInput( m_SubmodelCv[j]->GetClassSamples(i) );
        covarianceEstimator->Update();

        covarianceFold = covarianceEstimator->GetCovarianceMatrix().GetVnlMatrix();
        meanFold       = VectorType(covarianceEstimator->GetMean().GetDataPointer(),Superclass::m_FeatNb);

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


  // Precomputation of terms use for prediction //
  Superclass::m_Q.clear();
  Superclass::m_EigenValues.clear();
  Superclass::m_CstDecision.clear();
  Superclass::m_LambdaQ.clear();
  Superclass::m_Q.resize(Superclass::m_ClassNb,MatrixType(selectedVarNb,selectedVarNb));
  Superclass::m_EigenValues.resize(Superclass::m_ClassNb,VectorType(selectedVarNb));
  MatrixType subCovariance(selectedVarNb,selectedVarNb);

  Superclass::m_CstDecision.assign(Superclass::m_ClassNb,0);
  Superclass::m_LambdaQ.resize(Superclass::m_ClassNb, MatrixType(selectedVarNb,selectedVarNb));
  m_SubMeans.resize(Superclass::m_ClassNb, VectorType(selectedVarNb));

  RealType lambda;
  for ( unsigned int i = 0; i < Superclass::m_ClassNb; ++i )
  {
    // Decompose covariance matrix in eigenvalues/eigenvectors
    ExtractSubSymmetricMatrix(m_SelectedVar,Superclass::m_Covariances[i],subCovariance);
    Superclass::Decomposition(subCovariance, Superclass::m_Q[i], Superclass::m_EigenValues[i]);

    // Extract mean corresponding to slected variables
    ExtractVector(m_SelectedVar,Superclass::m_Means[i],m_SubMeans[i]);

    // Precompute lambda^(-1/2) * Q and log(det lambda)
    for (int j = 0; j < selectedVarNb; ++j)
    {
      lambda = 1 / sqrt(Superclass::m_EigenValues[i][j]);
      // Transposition and row multiplication at the same time
      Superclass::m_LambdaQ[i].set_row(j,lambda*Superclass::m_Q[i].get_column(j));

      Superclass::m_CstDecision[i] += log(Superclass::m_EigenValues[i][j]);
    }

    Superclass::m_CstDecision[i] += -2*log(Superclass::m_Proportion[i]);
  }
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ForwardSelection(std::string criterion, int selectedVarNb)
{
  // Initialization
  int currentSelectedVarNb = 0;
  RealType argMaxValue;
  std::vector<int> variablesPool;
  variablesPool.resize(Superclass::m_FeatNb);
  m_CriterionBestValues.resize(selectedVarNb);
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
      for (int i = 0; i < m_SubmodelCv.size(); ++i)
        m_SubmodelCv[i]->ComputeClassifRate(criterionVal,"forward",variablesPool,criterion);

      // Compute mean instead of keeping sum of criterion for all folds (not necessary)
      for (int i = 0; i < criterionVal.size(); ++i)
        criterionVal[i] /= m_SubmodelCv.size();
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
    m_CriterionBestValues[currentSelectedVarNb] = criterionVal[argMaxValue];

    // std::cout << "Criterion best values:";
    // for (typename std::vector<RealType>::iterator it = criterionBestValues.begin(); it != criterionBestValues.end(); ++it)
    // {
    //   std::cout << ' ' << *it;
    // }
    // std::cout << std::endl;

    // Add it to selected var and delete it from the pool
    m_SelectedVar.push_back(variablesPool[argMaxValue]);
    variablesPool.erase(variablesPool.begin()+argMaxValue);

    // Update submodel
    if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("F1mean") == 0) )
    {
      for (int i = 0; i < m_SubmodelCv.size(); ++i)
        m_SubmodelCv[i]->SetSelectedVar(m_SelectedVar,0);
    }

    currentSelectedVarNb++;
  }
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::FloatingForwardSelection(std::string criterion, int selectedVarNb)
{
  // Initialization
  int currentSelectedVarNb = 0;
  RealType argMaxValue;
  std::vector<int> variablesPool;
  variablesPool.resize(Superclass::m_FeatNb);
  m_CriterionBestValues.clear();
  m_SelectedVar.clear();
  for (int i = 0; i < Superclass::m_FeatNb; ++i)
    variablesPool[i] = i;

  std::vector<std::vector<int> > bestSets;
  std::vector<std::vector<int> > bestSetsPools;
  bool flagBacktrack;

  // Start the forward search
  while ((currentSelectedVarNb<selectedVarNb)&&(!variablesPool.empty()))
  {
    std::vector<RealType> criterionVal(variablesPool.size(),0);

    // Compute criterion function
    if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("F1mean") == 0) )
    {
      for (int i = 0; i < m_SubmodelCv.size(); ++i)
        m_SubmodelCv[i]->ComputeClassifRate(criterionVal,"forward",variablesPool,criterion);

      // Compute mean instead of keeping sum of criterion for all folds (not necessary)
      for (int i = 0; i < criterionVal.size(); ++i)
        criterionVal[i] /= m_SubmodelCv.size();
    }
    else if (criterion.compare("JM") == 0)
      ComputeJM(criterionVal,"forward",variablesPool);
    else if (criterion.compare("divKL") == 0)
      ComputeDivKL(criterionVal,"forward",variablesPool);

    // Select the variable that provides the highest criterion value
    argMaxValue = std::distance(criterionVal.begin(), std::max_element(criterionVal.begin(), criterionVal.end()));
    currentSelectedVarNb++;

    if ((currentSelectedVarNb <= m_CriterionBestValues.size()) && (criterionVal[argMaxValue] < m_CriterionBestValues[currentSelectedVarNb-1]))
    {
      m_SelectedVar = bestSets[currentSelectedVarNb-1];
      variablesPool = bestSetsPools[currentSelectedVarNb-1];
    }
    else
    {
      // Add it to selected var and delete it from the pool
      m_SelectedVar.push_back(variablesPool[argMaxValue]);
      variablesPool.erase(variablesPool.begin()+argMaxValue);

      // Update submodel
      if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("F1mean") == 0) )
      {
        for (int i = 0; i < m_SubmodelCv.size(); ++i)
          m_SubmodelCv[i]->SetSelectedVar(m_SelectedVar,0);
      }

      if (currentSelectedVarNb > m_CriterionBestValues.size())
      {
        m_CriterionBestValues.push_back(criterionVal[argMaxValue]);
        bestSets.push_back(m_SelectedVar);
        bestSetsPools.push_back(variablesPool);
      }
      else
      {
        m_CriterionBestValues[currentSelectedVarNb-1] = criterionVal[argMaxValue];
        bestSets[currentSelectedVarNb-1] = m_SelectedVar;
        bestSetsPools[currentSelectedVarNb-1] = variablesPool;
      }

      flagBacktrack = true;

      while (flagBacktrack && (currentSelectedVarNb > 2))
      {

        std::vector<RealType> criterionValBackward(m_SelectedVar.size(),0);

        // Compute criterion function
        if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("F1mean") == 0) )
        {
          for (int i = 0; i < m_SubmodelCv.size(); ++i)
            m_SubmodelCv[i]->ComputeClassifRate(criterionValBackward,"backward",m_SelectedVar,criterion);

          // Compute mean instead of keeping sum of criterion for all folds (not necessary)
          for (int i = 0; i < criterionValBackward.size(); ++i)
            criterionValBackward[i] /= m_SubmodelCv.size();
        }
        else if (criterion.compare("JM") == 0)
          ComputeJM(criterionValBackward,"backward",m_SelectedVar);
        else if (criterion.compare("divKL") == 0)
          ComputeDivKL(criterionValBackward,"backward",m_SelectedVar);

        argMaxValue = std::distance(criterionValBackward.begin(), std::max_element(criterionValBackward.begin(), criterionValBackward.end()));

        if (criterionValBackward[argMaxValue] > m_CriterionBestValues[currentSelectedVarNb-2])
        {
          currentSelectedVarNb--;

          variablesPool.push_back(m_SelectedVar[argMaxValue]);
          m_SelectedVar.erase(m_SelectedVar.begin()+argMaxValue);

          m_CriterionBestValues[currentSelectedVarNb-1] = criterionValBackward[argMaxValue];
          bestSets[currentSelectedVarNb-1] = m_SelectedVar;
          bestSetsPools[currentSelectedVarNb-1] = variablesPool;
        }
        else
          flagBacktrack = false;
      }
    }
  }
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ComputeClassifRate(std::vector<RealType> & criterionVal, const std::string direction, std::vector<int> & variablesPool, const std::string criterion)
{
  typedef ConfusionMatrixCalculator< TargetListSampleType, TargetListSampleType > ConfusionMatrixType;

  if (m_SelectedVar.empty())
  {
    InputSampleType sample;
    TargetSampleType res;
    std::vector<RealType> scores(Superclass::m_ClassNb);

    for (int k = 0; k < variablesPool.size(); ++k)
    {
      typename TargetListSampleType::Pointer TargetListSample    = TargetListSampleType::New();
      typename TargetListSampleType::Pointer RefTargetListSample = TargetListSampleType::New();
      typename ConfusionMatrixType::Pointer confM = ConfusionMatrixType::New();

      for (int i = 0; i < m_Fold.size(); ++i)
      {
        for (int j = 0; j < Superclass::m_NbSpl[i]; ++j)
        {
          sample = m_Fold[i]->GetMeasurementVectorByIndex(j);

          for (int c = 0; c < Superclass::m_ClassNb; ++c)
            scores[c] = (sample[k] - Superclass::m_Means[c][k])*(sample[k] - Superclass::m_Means[c][k]) / Superclass::m_Covariances[c](k,k) + log(Superclass::m_Covariances[c](k,k)) - m_Logprop[c];

          res[0] = Superclass::m_MapOfIndices.at(std::distance(scores.begin(), std::min_element(scores.begin(), scores.end())));
          TargetListSample->PushBack(res);
          res[0] = Superclass::m_MapOfIndices.at(i);
          RefTargetListSample->PushBack(res);
        }
      }

      confM->SetReferenceLabels(RefTargetListSample);
      confM->SetProducedLabels(TargetListSample);
      confM->Compute();

      if (criterion.compare("accuracy") == 0)
      {
        criterionVal[k] += (RealType) confM->GetOverallAccuracy();
      }
      else if (criterion.compare("kappa") == 0)
      {
        criterionVal[k] += (RealType) confM->GetKappaIndex();
      }
      else if (criterion.compare("F1mean") == 0)
      {
        typename ConfusionMatrixType::MeasurementType Fscores = confM->GetFScores();
        RealType meanFscores = 0;
        for (int i = 0; i < Fscores.Size(); ++i)
          meanFscores += (RealType) Fscores[i];
        criterionVal[k] += meanFscores/Superclass::m_ClassNb;
      }
    }
  }
  else
  {
    // Get info
    int selectedVarNb = m_SelectedVar.size();

    // Allocation
    MatrixType subCovariances(selectedVarNb,selectedVarNb);
    MatrixType Q(selectedVarNb,selectedVarNb);
    std::vector<MatrixType> invCov(Superclass::m_ClassNb,MatrixType(selectedVarNb,selectedVarNb));
    std::vector<MatrixType> subMeans(Superclass::m_ClassNb,MatrixType(selectedVarNb,1));
    VectorType eigenValues(selectedVarNb);
    std::vector<RealType> logdet(Superclass::m_ClassNb);

    // Compute inv of covariance matrix
    for (int c = 0; c < Superclass::m_ClassNb; ++c)
    {
      ExtractVectorToColMatrix(m_SelectedVar, Superclass::m_Means[c], subMeans[c]);
      ExtractSubSymmetricMatrix(m_SelectedVar,Superclass::m_Covariances[c],subCovariances);
      Superclass::Decomposition(subCovariances, Q, eigenValues);

      invCov[c] = Q * (vnl_diag_matrix<RealType>(eigenValues).invert_in_place() * Q.transpose());
      logdet[c] = eigenValues.apply(log).sum();
    }

    InputSampleType sample;
    TargetSampleType res;
    std::vector<RealType> scores(Superclass::m_ClassNb);
    std::vector<RealType> alpha(Superclass::m_ClassNb);
    std::vector<RealType> logdet_update(Superclass::m_ClassNb);
    std::vector<MatrixType> v(Superclass::m_ClassNb,MatrixType(selectedVarNb,1));
    MatrixType u(selectedVarNb,1);
    VectorType input(Superclass::m_FeatNb);
    MatrixType subInput(selectedVarNb,1);

    for (int k = 0; k < variablesPool.size(); ++k)
    {
      typename TargetListSampleType::Pointer TargetListSample    = TargetListSampleType::New();
      typename TargetListSampleType::Pointer RefTargetListSample = TargetListSampleType::New();
      typename ConfusionMatrixType::Pointer confM = ConfusionMatrixType::New();

      if (direction.compare("forward")==0)
      {
        for (int c = 0; c < Superclass::m_ClassNb; ++c)
        {
          ExtractReducedColumn(variablesPool[k],m_SelectedVar,Superclass::m_Covariances[c],u);
          alpha[c] = Superclass::m_Covariances[c](variablesPool[k],variablesPool[k]) - (u.transpose() * invCov[c] *u)(0,0);
          if (alpha[c] < std::numeric_limits<RealType>::epsilon())
            alpha[c] = std::numeric_limits<RealType>::epsilon();

          logdet_update[c] = log(alpha[c]) + logdet[c];
          v[c] = -1/alpha[c] * (invCov[c]*u);
        }

        for (int i = 0; i < m_Fold.size(); ++i)
        {
          for (int j = 0; j < Superclass::m_NbSpl[i]; ++j)
          {
            sample = m_Fold[i]->GetMeasurementVectorByIndex(j);

            // Convert input data
            vnl_copy(vnl_vector<InputValueType>(sample.GetDataPointer(), Superclass::m_FeatNb),input);
            for (int n = 0; n < selectedVarNb; ++n)
              subInput(n,0) = input[m_SelectedVar[n]];

            for (int c = 0; c < Superclass::m_ClassNb; ++c)
              scores[c] =  ((subInput.transpose() - subMeans[c])*invCov[c]*(subInput - subMeans[c]))(0,0) + alpha[c]*pow(((subInput - subMeans[c])*v[c])(0,0) + 1/alpha[c] * (input[variablesPool[k]] - Superclass::m_Means[c][variablesPool[k]]),2) + logdet_update[c] - m_Logprop[c];

            res[0] = Superclass::m_MapOfIndices.at(std::distance(scores.begin(), std::min_element(scores.begin(), scores.end())));
            TargetListSample->PushBack(res);
            res[0] = Superclass::m_MapOfIndices.at(i);
            RefTargetListSample->PushBack(res);
          }
        }
      }
      else if (direction.compare("backward")==0)
      {
        for (int c = 0; c < Superclass::m_ClassNb; ++c)
        {
          alpha[c] = 1/invCov[c](k,k);
          if (alpha[c] < std::numeric_limits<RealType>::epsilon())
            alpha[c] = std::numeric_limits<RealType>::epsilon();

          logdet_update[c] = logdet[c] - log(alpha[c]);
          v[c] = invCov[c].get_n_columns(k,1);
        }

        for (int i = 0; i < m_Fold.size(); ++i)
        {
          for (int j = 0; j < Superclass::m_NbSpl[i]; ++j)
          {
            sample = m_Fold[i]->GetMeasurementVectorByIndex(j);

            // Convert input data
            vnl_copy(vnl_vector<InputValueType>(sample.GetDataPointer(), Superclass::m_FeatNb),input);
            for (int n = 0; n < selectedVarNb; ++n)
              subInput(n,0) = input[m_SelectedVar[n]];

            std::vector<RealType> scores(Superclass::m_ClassNb);
            for (int c = 0; c < Superclass::m_ClassNb; ++c)
              scores[c] =  ((subInput.transpose() - subMeans[c])*invCov[c]*(subInput - subMeans[c]))(0,0)  - alpha[c]*pow((v[c].transpose()*(subInput - subMeans[c]))(0,0),2) + logdet_update[c] - m_Logprop[c];

            res[0] = Superclass::m_MapOfIndices.at(std::distance(scores.begin(), std::min_element(scores.begin(), scores.end())));
            TargetListSample->PushBack(res);
            res[0] = Superclass::m_MapOfIndices.at(i);
            RefTargetListSample->PushBack(res);
          }
        }
      }

      confM->SetReferenceLabels(RefTargetListSample);
      confM->SetProducedLabels(TargetListSample);
      confM->Compute();

      if (criterion.compare("accuracy") == 0)
      {
        criterionVal[k] += (RealType) confM->GetOverallAccuracy();
      }
      else if (criterion.compare("kappa") == 0)
      {
        criterionVal[k] += (RealType) confM->GetKappaIndex();
      }
      else if (criterion.compare("F1mean") == 0)
      {
        typename ConfusionMatrixType::MeasurementType Fscores = confM->GetFScores();
        RealType meanFscores = 0;
        for (int i = 0; i < Fscores.Size(); ++i)
          meanFscores += (RealType) Fscores[i];
        criterionVal[k] += meanFscores/Superclass::m_ClassNb;
      }
    }
  }
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
        for (int k = 0; k < variablesPool.size(); ++k)
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
            alpha = 1/invCov(k,k);
            if (alpha < std::numeric_limits<RealType>::epsilon())
              alpha = std::numeric_limits<RealType>::epsilon();

            logdet_c1c2 = logdet - log(alpha) + (selectedVarNb-1)*log(2);

            cst_feat = - alpha * pow( (invCov.get_n_rows(k,1)*md)(0,0), 2);
          }

          bij = (1/8.) * (md.transpose() * (invCov*md))(0,0) + cst_feat/8 + 0.5*(logdet_c1c2 - halfedLogdet[c1][k] - halfedLogdet[c2][k]);
          JM[k] += Superclass::m_Proportion[c1] * Superclass::m_Proportion[c2] * sqrt(2*(1-exp(-bij)));
        }
      }
    }
  }
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ComputeDivKL(std::vector<RealType> & divKL, const std::string direction, std::vector<int> & variablesPool)
{
  // Get info
  int selectedVarNb = m_SelectedVar.size();


  if (m_SelectedVar.empty())
  {
    RealType alpha1,alpha2;

    for (int k = 0; k < variablesPool.size(); ++k)
    {
      for (int c1 = 0; c1 < Superclass::m_ClassNb; ++c1)
      {
        alpha1 = 1/Superclass::m_Covariances[c1](variablesPool[k],variablesPool[k]);
        if (alpha1 < std::numeric_limits<RealType>::epsilon())
          alpha1 = std::numeric_limits<RealType>::epsilon();

        for (int c2 = c1+1; c2 < Superclass::m_ClassNb; ++c2)
        {
          alpha2 = 1/Superclass::m_Covariances[c2](variablesPool[k],variablesPool[k]);
          if (alpha2 < std::numeric_limits<RealType>::epsilon())
            alpha2 = std::numeric_limits<RealType>::epsilon();

          divKL[k] += Superclass::m_Proportion[c1] * Superclass::m_Proportion[c2] * 0.5 * (alpha1*Superclass::m_Covariances[c2](variablesPool[k],variablesPool[k]) + alpha2*Superclass::m_Covariances[c1](variablesPool[k],variablesPool[k]) + (Superclass::m_Means[c1](variablesPool[k]) - Superclass::m_Means[c2](variablesPool[k]))*(alpha1+alpha2)*(Superclass::m_Means[c1](variablesPool[k]) - Superclass::m_Means[c2](variablesPool[k])));
        }
      }
    }
  }
  else
  {
    // Allocation
    MatrixType reducedCovariances(selectedVarNb,selectedVarNb);
    MatrixType Q(selectedVarNb,selectedVarNb);
    VectorType eigenValues(selectedVarNb);
    std::vector<MatrixType> invCov(Superclass::m_ClassNb,MatrixType(selectedVarNb,selectedVarNb));
    int newVarNb;

    if (direction.compare("forward")==0)
      newVarNb = selectedVarNb + 1;
    else if (direction.compare("backward")==0)
      newVarNb = selectedVarNb - 1;
    std::vector<MatrixType> invCov_update(Superclass::m_ClassNb,MatrixType(newVarNb,newVarNb));

    for (int c = 0; c < Superclass::m_ClassNb; ++c)
    {
      ExtractSubSymmetricMatrix(m_SelectedVar,Superclass::m_Covariances[c],reducedCovariances);
      Superclass::Decomposition(reducedCovariances, Q, eigenValues);
      invCov[c] = Q * (vnl_diag_matrix<RealType>(eigenValues).invert_in_place() * Q.transpose());
    }

    RealType alpha;
    MatrixType tmp(selectedVarNb,selectedVarNb);
    MatrixType subMatrix(selectedVarNb-1,selectedVarNb-1);
    std::vector<MatrixType> subMeans(Superclass::m_ClassNb,MatrixType(newVarNb,1));
    MatrixType u(selectedVarNb,1);
    MatrixType md(newVarNb,1);
    std::vector<int> newSelectedVar(newVarNb);
    for (int k = 0; k < variablesPool.size(); ++k)
    {
      if (direction.compare("forward")==0)
      {
        std::vector<int>::iterator varIt = m_SelectedVar.begin();
        for (int i = 0; i < selectedVarNb; ++i)
        {
          newSelectedVar[i] = *varIt;
          varIt++;
        }
        newSelectedVar[newVarNb-1] = variablesPool[k];

        for (int c = 0; c < Superclass::m_ClassNb; ++c)
        {
          ExtractReducedColumn(variablesPool[k],m_SelectedVar,Superclass::m_Covariances[c],u);
          tmp = invCov[c]*u;

          alpha = Superclass::m_Covariances[c](variablesPool[k],variablesPool[k]) - (u.transpose()*tmp)(0,0);
          if (alpha < std::numeric_limits<RealType>::epsilon())
            alpha = std::numeric_limits<RealType>::epsilon();

          invCov_update[c].update(invCov[c] + (1/alpha) * tmp*tmp.transpose(),0,0);
          invCov_update[c].update(-(1/alpha) * tmp,0,newVarNb-1);
          invCov_update[c].update(-(1/alpha) * tmp.transpose(),newVarNb-1,0);
          invCov_update[c](newVarNb-1,newVarNb-1) = 1/alpha;
        }
      }
      else if (direction.compare("backward")==0)
      {
        std::vector<int>::iterator varIt = newSelectedVar.begin();
        for (int i = 0; i < selectedVarNb; ++i)
        {
          if (i!=k)
          {
            *varIt = m_SelectedVar[i];
            varIt++;
          }
        }

        for (int c = 0; c < Superclass::m_ClassNb; ++c)
        {
          ExtractSubSymmetricMatrix(newSelectedVar,Superclass::m_Covariances[c],subMatrix);
          ExtractReducedColumn(variablesPool[k],newSelectedVar,Superclass::m_Covariances[c],u);
          invCov_update[c] = subMatrix - 1/invCov[c](k,k) * u * u.transpose();
        }
      }

      // Extract means
      std::vector<MatrixType> subCovariances(Superclass::m_ClassNb,MatrixType(newVarNb,newVarNb));
      for (int c = 0; c < Superclass::m_ClassNb; ++c)
      {
        ExtractVectorToColMatrix(newSelectedVar, Superclass::m_Means[c], subMeans[c]);
        ExtractSubSymmetricMatrix(newSelectedVar,Superclass::m_Covariances[c],subCovariances[c]);
      }

      for (int c1 = 0; c1 < Superclass::m_ClassNb; ++c1)
      {
        for (int c2 = c1+1; c2 < Superclass::m_ClassNb; ++c2)
        {
          md       = subMeans[c1] - subMeans[c2];
          divKL[k] += Superclass::m_Proportion[c1] * Superclass::m_Proportion[c2] * 0.5 * ( vnl_trace(invCov_update[c2]*subCovariances[c1] + invCov_update[c1]*subCovariances[c2]) + (md.transpose()*(invCov_update[c1]+invCov_update[c2])*md)(0,0) );
        }
      }
    }
  }
}

/** Train the machine learning model */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Train()
{
  Superclass::Train();
}

template <class TInputValue, class TTargetValue>
typename GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::TargetSampleType
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Predict(const InputSampleType & rawInput, ConfidenceValueType *quality) const
{
  if (quality != NULL)
  {
    if (!this->HasConfidenceIndex())
    {
      itkExceptionMacro("Confidence index not available for this classifier !");
    }
  }

  if (m_SelectedVar.empty())
  {
    return Superclass::Predict(rawInput, quality);
  }
  else
  {
    // Convert input data
    VectorType input(Superclass::m_FeatNb);
    VectorType subInput(m_SelectedVar.size());
    vnl_copy(vnl_vector<InputValueType>(rawInput.GetDataPointer(), Superclass::m_FeatNb),input);

    for (int i = 0; i < m_SelectedVar.size(); ++i)
      subInput[i] = input[m_SelectedVar[i]];

    // Compute decision function
    std::vector<RealType> decisionFct(Superclass::m_CstDecision);
    VectorType lambdaQInputC(m_SelectedVar.size());
    VectorType input_c(m_SelectedVar.size());
    for (int i = 0; i < Superclass::m_ClassNb; ++i)
    {
      input_c = subInput - m_SubMeans[i];
      lambdaQInputC = Superclass::m_LambdaQ[i] * input_c;

      // Add sum of squared elements
      decisionFct[i] += lambdaQInputC.squared_magnitude();
    }

    int argmin = std::distance(decisionFct.begin(), std::min_element(decisionFct.begin(), decisionFct.end()));

    TargetSampleType res;
    res[0] = Superclass::m_MapOfIndices.at(argmin);

    return res;
  }
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

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::SetSelectedVar(std::vector<int> varSubSet, int recompute)
{
  m_SelectedVar = varSubSet;

  if (recompute == 1)
  {
    int selectedVarNb = m_SelectedVar.size();

    // Precomputation of terms use for prediction //
    Superclass::m_Q.clear();
    Superclass::m_EigenValues.clear();
    Superclass::m_CstDecision.clear();
    Superclass::m_LambdaQ.clear();
    Superclass::m_Q.resize(Superclass::m_ClassNb,MatrixType(selectedVarNb,selectedVarNb));
    Superclass::m_EigenValues.resize(Superclass::m_ClassNb,VectorType(selectedVarNb));
    MatrixType subCovariance(selectedVarNb,selectedVarNb);

    Superclass::m_CstDecision.resize(Superclass::m_ClassNb,0);
    Superclass::m_LambdaQ.resize(Superclass::m_ClassNb, MatrixType(selectedVarNb,selectedVarNb));
    m_SubMeans.resize(Superclass::m_ClassNb, VectorType(selectedVarNb));

    RealType lambda;
    for ( unsigned int i = 0; i < Superclass::m_ClassNb; ++i )
    {
      // Decompose covariance matrix in eigenvalues/eigenvectors
      ExtractSubSymmetricMatrix(m_SelectedVar,Superclass::m_Covariances[i],subCovariance);
      Superclass::Decomposition(subCovariance, Superclass::m_Q[i], Superclass::m_EigenValues[i]);

      // Extract mean corresponding to slected variables
      ExtractVector(m_SelectedVar,Superclass::m_Means[i],m_SubMeans[i]);

      // Precompute lambda^(-1/2) * Q and log(det lambda)
      for (int j = 0; j < selectedVarNb; ++j)
      {
        lambda = 1 / sqrt(Superclass::m_EigenValues[i][j]);
        // Transposition and row multiplication at the same time
        Superclass::m_LambdaQ[i].set_row(j,lambda*Superclass::m_Q[i].get_column(j));

        Superclass::m_CstDecision[i] += log(Superclass::m_EigenValues[i][j]);
      }

      Superclass::m_CstDecision[i] += -2*log(Superclass::m_Proportion[i]);
    }
  }
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Save(const std::string & filename, const std::string & name)
{
  Superclass::Save(filename,name);

  if (m_SelectedVar.size() != 0)
  {
    std::string selectionFilename = "selection_" + filename;
    std::ofstream ofs(selectionFilename.c_str(), std::ios::out);

    // Store number of selected variables
    ofs << m_SelectedVar.size() << std::endl;

    // Store length of subMean vectors
    ofs << m_SubMeans[0].size() << std::endl;

    // Store vector of selected features
    for (int i = 0; i < m_SelectedVar.size(); ++i)
      ofs << m_SelectedVar[i] << " ";
    ofs << std::endl;

    // Store vector of criterion functions values with the corresponding number of features used
    for (int i = 0; i < m_SelectedVar.size(); ++i)
      ofs << i+1 << " " << m_CriterionBestValues[i] << std::endl;

    // Store vector of size C containing the mean vector of each class for the selected variables (one by line)
    for (int i = 0; i < Superclass::m_ClassNb; ++i)
    {
      for (int j = 0; j < m_SubMeans[i].size(); ++j)
        ofs << m_SubMeans[i][j] << " ";

      ofs << std::endl;
    }

    ofs.close();
  }
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Load(const std::string & filename, const std::string & name)
{
  Superclass::Load(filename,name);

  std::string selectionFilename = "selection_" + filename;
  std::ifstream ifs(selectionFilename.c_str(), std::ios::in);

  if(!ifs)
  {
    std::cerr<<"Could not found/read file "<<selectionFilename<<std::endl;
  }
  else
  {
    int selectedVarNb, subMeanSize, dump;
    ifs >> selectedVarNb;
    ifs >> subMeanSize;

    // Allocation
    m_SelectedVar.resize(selectedVarNb);
    m_CriterionBestValues.resize(selectedVarNb);
    m_Logprop.resize(Superclass::m_ClassNb);
    m_SubMeans.resize(Superclass::m_ClassNb,VectorType(subMeanSize));

    // Load selected variables
    for (int i = 0; i < selectedVarNb; ++i)
      ifs >> m_SelectedVar[i];

    // Load criterion function values
    for (int i = 0; i < selectedVarNb; ++i)
    {
      ifs >> dump;
      ifs >> m_CriterionBestValues[i];
    }

    // Load subMean
    for (int i = 0; i < Superclass::m_ClassNb; ++i)
      for (int j = 0; j < subMeanSize; ++j)
        ifs >> m_SubMeans[i][j];
  }

  ifs.close();
}

} //end namespace otb

#endif
