#ifndef __otbGMMSelectionMachineLearningModel_txx
#define __otbGMMSelectionMachineLearningModel_txx

#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include <vector>
#include <algorithm>

#include "otbGMMSelectionMachineLearningModel.h"

#include "otbConfusionMatrixCalculator.h"
#include "vnl/vnl_trace.h"

namespace otb
{

template <class TInputValue, class TTargetValue>
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::GMMSelectionMachineLearningModel():
  m_EnableOptimalSet(true)
{
}

template <class TInputValue, class TTargetValue>
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::~GMMSelectionMachineLearningModel()
{
}

/***************************************/
/********   Extract submatrix   ********/
/***************************************/

/** Extract a vector from a vector by indexes */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ExtractVector(const std::vector<int> & indexes, const VectorType& input, VectorType& output)
{
  typename VectorType::iterator outIt = output.begin();
  for (std::vector<int>::const_iterator it = indexes.begin(); it != indexes.end(); ++it, ++outIt)
    *outIt = input[*it];
}

/** Extract a column matrix from a vector by indexes */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ExtractVectorToColMatrix(const std::vector<int> & indexes, const VectorType& input, MatrixType& output)
{
  std::vector<int>::const_iterator it = indexes.begin();
  for (unsigned int i = 0; i < indexes.size(); ++i, ++it)
    output(i,0) = input[*it];
}

/** Extract a column matrix from a matrix by column nb and indexes */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ExtractReducedColumn(const int colIndex, const std::vector<int> & indexesRow, const MatrixType& input, MatrixType& output)
{
  std::vector<int>::const_iterator it = indexesRow.begin();
  for (unsigned int i = 0; i < indexesRow.size(); ++i, ++it)
    output(i,0) = input(*it,colIndex);
}

/** Extract a matrix from a symmetric matrix by indexes */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ExtractSubSymmetricMatrix(const std::vector<int> & indexes, const MatrixType& input, MatrixType& output)
{
  for (unsigned int i = 0; i < indexes.size(); ++i)
  {
    output(i,i) = input(indexes[i],indexes[i]);
    for (unsigned int j = i+1; j < indexes.size(); ++j)
    {
      output(i,j) = input(indexes[i],indexes[j]);
      output(j,i) = output(i,j);
    }
  }
}

/***************************************/
/************     Tools    *************/
/***************************************/

/** Add samples to fold */
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

/** Update model (proportion and cst 2log(prop)) */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::UpdateProportion()
{
  unsigned totalNb = 0;
  for (std::vector<unsigned>::iterator it = Superclass::m_NbSpl.begin(); it != Superclass::m_NbSpl.end(); ++it)
    totalNb += *it;

  Superclass::m_Proportion.resize(Superclass::m_ClassNb);
  m_Logprop.resize(Superclass::m_ClassNb);
  for (unsigned int i = 0; i < Superclass::m_ClassNb; ++i)
  {
    Superclass::m_Proportion[i] = (double) Superclass::m_NbSpl[i] / (double) totalNb;
    m_Logprop[i]                = 2 * (RealType) log(Superclass::m_Proportion[i]);
  }
}

/***************************************/
/******   Selection algorithm    *******/
/***************************************/

/** Front-end function to call selection */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Selection(std::string direction, std::string criterion, int selectedVarNb, int nfold, int seed)
{
  // Creation of submodel for cross-validation
  if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("f1mean") == 0))
  {
    // Allocation
    m_SubmodelCv.resize(nfold);
    for (int j = 0; j < nfold; ++j)
      m_SubmodelCv[j] = GMMSelectionMachineLearningModel<TInputValue, TTargetValue>::New();
    typedef itk::Statistics::CovarianceSampleFilter< itk::Statistics::Subsample< InputListSampleType > > CovarianceEstimatorType;
    typename CovarianceEstimatorType::Pointer covarianceEstimator = CovarianceEstimatorType::New();
    VectorType meanFold;
    MatrixType covarianceFold, adjustedMean;
    unsigned nbSplFold;

    for (unsigned int i = 0; i < Superclass::m_ClassNb; ++i)
    {
      // Get all sample ids from ith class
      std::vector<InstanceIdentifier> indices(Superclass::m_NbSpl[i]);
      for (unsigned j=0; j<Superclass::m_NbSpl[i]; ++j)
        indices[j] = (Superclass::m_ClassSamples[i])->GetInstanceIdentifier(j);

      // Shuffle id of samples
      std::srand( unsigned( seed ) );
      std::random_shuffle( indices.begin(), indices.end() );

      nbSplFold = Superclass::m_NbSpl[i]/nfold;

      for (int j = 0; j < nfold; ++j)
      {
        // Add subpart of id to fold
        if (j==nfold-1)
        {
          m_SubmodelCv[j]->AddInstanceToFold(Superclass::GetInputListSample(), indices,j*nbSplFold,Superclass::m_NbSpl[i]);
          m_SubmodelCv[j]->AddNbSpl(Superclass::m_NbSpl[i] - j*nbSplFold);
          nbSplFold = Superclass::m_NbSpl[i] - j*nbSplFold;
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

        m_SubmodelCv[j]->AddMean( (1/(RealType)(Superclass::m_NbSpl[i] - nbSplFold)) * ((RealType) Superclass::m_NbSpl[i] * Superclass::m_Means[i] - (RealType) nbSplFold * meanFold) );
        adjustedMean = MatrixType((Superclass::m_Means[i]-meanFold).data_block(), Superclass::m_FeatNb, 1); // convert to matrix
        m_SubmodelCv[j]->AddCovMatrix( (1/(RealType)(Superclass::m_NbSpl[i]-nbSplFold-1)) * ( ((RealType)Superclass::m_NbSpl[i]-1)*Superclass::m_Covariances[i] - ((RealType)nbSplFold-1)*covarianceFold - (RealType)Superclass::m_NbSpl[i]*(RealType)nbSplFold/(RealType)(Superclass::m_NbSpl[i]-nbSplFold) * adjustedMean * adjustedMean.transpose() ) ); // convert all unsigned in realType
      }
    }

    for (int i = 0; i < nfold; ++i)
      m_SubmodelCv[i]->UpdateProportion();
  }

  // Perform selection
  if (direction.compare("forward") == 0)
    ForwardSelection(criterion, selectedVarNb);
  else if (direction.compare("sffs") == 0)
    FloatingForwardSelection(criterion, selectedVarNb);

  // Choose set for prediction (best set or full set)
  std::vector<int> varPrediction;
  if (m_EnableOptimalSet)
  {
    m_VarNbPrediction = 1 + std::distance(m_CriterionBestValues.begin(), std::max_element(m_CriterionBestValues.begin(), m_CriterionBestValues.end()));
    for (int i = 0; i < m_VarNbPrediction; ++i)
      varPrediction.push_back(m_SelectedVar[i]);
  }
  else
  {
    m_VarNbPrediction = selectedVarNb;
    varPrediction     = m_SelectedVar;
  }

  // Precomputation of terms use for prediction
  std::vector<MatrixType>().swap(Superclass::m_Q);
  std::vector<MatrixType>().swap(Superclass::m_LambdaQ);
  std::vector<VectorType>().swap(Superclass::m_EigenValues);
  std::vector<RealType>().swap(Superclass::m_CstDecision);

  Superclass::m_Q.resize(Superclass::m_ClassNb,MatrixType(m_VarNbPrediction,m_VarNbPrediction));
  Superclass::m_EigenValues.resize(Superclass::m_ClassNb,VectorType(m_VarNbPrediction));
  MatrixType subCovariance(m_VarNbPrediction,m_VarNbPrediction);
  Superclass::m_CstDecision.assign(Superclass::m_ClassNb,0);
  Superclass::m_LambdaQ.resize(Superclass::m_ClassNb, MatrixType(m_VarNbPrediction,m_VarNbPrediction));
  m_SubMeans.resize(Superclass::m_ClassNb, VectorType(m_VarNbPrediction));

  RealType lambda;
  for ( unsigned int i = 0; i < Superclass::m_ClassNb; ++i )
  {
    // Decompose covariance matrix in eigenvalues/eigenvectors
    ExtractSubSymmetricMatrix(varPrediction,Superclass::m_Covariances[i],subCovariance);
    Superclass::Decomposition(subCovariance, Superclass::m_Q[i], Superclass::m_EigenValues[i]);

    // Extract mean corresponding to slected variables
    ExtractVector(varPrediction,Superclass::m_Means[i],m_SubMeans[i]);

    // Precompute lambda^(-1/2) * Q and log(det lambda)
    for (int j = 0; j < m_VarNbPrediction; ++j)
    {
      lambda = 1 / sqrt(Superclass::m_EigenValues[i][j]);
      // Transposition and row multiplication at the same time
      Superclass::m_LambdaQ[i].set_row(j,lambda*Superclass::m_Q[i].get_column(j));

      Superclass::m_CstDecision[i] += log(Superclass::m_EigenValues[i][j]);
    }

    Superclass::m_CstDecision[i] += -2*log(Superclass::m_Proportion[i]);
  }
}

/** Perform sequential forward selection */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ForwardSelection(std::string criterion, int selectedVarNb)
{
  // Initialization
  int currentSelectedVarNb = 0;
  RealType argMaxValue;
  m_CriterionBestValues.resize(selectedVarNb);
  m_SelectedVar.clear();
  std::vector<int> variablesPool(Superclass::m_FeatNb);
  for (unsigned i = 0; i < Superclass::m_FeatNb; ++i)
    variablesPool[i] = i;

  // Start the forward search
  while ((currentSelectedVarNb<selectedVarNb)&&(!variablesPool.empty()))
  {
    std::vector<RealType> criterionVal(variablesPool.size(),0);

    // Compute criterion function
    if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("f1mean") == 0) )
    {
      // COULD BE PARALLELIZED but need to see if it is better to parallelize here on each fold or inside ComputeClassifRate
      for (unsigned int i = 0; i < m_SubmodelCv.size(); ++i)
        m_SubmodelCv[i]->ComputeClassifRate(criterionVal,"forward",variablesPool,criterion);

      // Compute mean instead of keeping sum of criterion for all folds (not necessary)
      for (unsigned int i = 0; i < criterionVal.size(); ++i)
        criterionVal[i] /= m_SubmodelCv.size();
    }
    else if (criterion.compare("jm") == 0)
    {
      ComputeJM(criterionVal,"forward",variablesPool);
    }
    else if (criterion.compare("divkl") == 0)
    {
      ComputeDivKL(criterionVal,"forward",variablesPool);
    }

    // Select the variable that provides the highest criterion value
    argMaxValue = std::distance(criterionVal.begin(), std::max_element(criterionVal.begin(), criterionVal.end()));
    m_CriterionBestValues[currentSelectedVarNb] = criterionVal[argMaxValue];

    // Add it to selected var and delete it from the pool
    m_SelectedVar.push_back(variablesPool[argMaxValue]);
    variablesPool.erase(variablesPool.begin()+argMaxValue);

    // Update submodel
    if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("f1mean") == 0) )
    {
      for (unsigned i = 0; i < m_SubmodelCv.size(); ++i)
        m_SubmodelCv[i]->SetSelectedVar(m_SelectedVar,0);
    }

    currentSelectedVarNb++;
  }

  // To be compatible with SFFS and save
  m_BestSets.resize(m_SelectedVar.size());
  for (unsigned i = 0; i < m_SelectedVar.size(); ++i)
  {
    std::vector<int> v(m_SelectedVar.begin(),m_SelectedVar.begin()+i+1);
    m_BestSets[i] = v;
  }
}

/** Perform sequential floating forward selection */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::FloatingForwardSelection(std::string criterion, int selectedVarNb)
{
  // Initialization
  int currentSelectedVarNb = 0;
  RealType argMaxValue;
  m_CriterionBestValues.clear();
  m_SelectedVar.clear();
  std::vector<int> variablesPool(Superclass::m_FeatNb);
  for (unsigned i = 0; i < Superclass::m_FeatNb; ++i)
    variablesPool[i] = i;

  m_BestSets.clear();
  std::vector<std::vector<int> > bestSetsPools;
  bool flagBacktrack;

  // Start the forward search
  while ((currentSelectedVarNb<selectedVarNb) && (!variablesPool.empty()))
  {
    std::vector<RealType> criterionVal(variablesPool.size(),0);

    // Compute criterion function
    if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("f1mean") == 0) )
    {
      // COULD BE PARALLELIZED but need to see if it is better to parallelize here on each fold or inside ComputeClassifRate
      for (unsigned int i = 0; i < m_SubmodelCv.size(); ++i)
        m_SubmodelCv[i]->ComputeClassifRate(criterionVal,"forward",variablesPool,criterion);

      // Compute mean instead of keeping sum of criterion for all folds (not necessary)
      for (unsigned int i = 0; i < criterionVal.size(); ++i)
        criterionVal[i] /= m_SubmodelCv.size();
    }
    else if (criterion.compare("jm") == 0)
      ComputeJM(criterionVal,"forward",variablesPool);
    else if (criterion.compare("divkl") == 0)
      ComputeDivKL(criterionVal,"forward",variablesPool);

    // Select the variable that provides the highest criterion value
    argMaxValue = std::distance(criterionVal.begin(), std::max_element(criterionVal.begin(), criterionVal.end()));
    currentSelectedVarNb++;

    if ((currentSelectedVarNb <= (int)m_CriterionBestValues.size()) && (criterionVal[argMaxValue] < m_CriterionBestValues[currentSelectedVarNb-1]))
    {
      m_SelectedVar = m_BestSets[currentSelectedVarNb-1];
      variablesPool = bestSetsPools[currentSelectedVarNb-1];
    }
    else
    {
      // Add it to selected var and delete it from the pool
      m_SelectedVar.push_back(variablesPool[argMaxValue]);
      variablesPool.erase(variablesPool.begin()+argMaxValue);

      // Update submodel
      if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("f1mean") == 0) )
      {
        for (unsigned i = 0; i < m_SubmodelCv.size(); ++i)
          m_SubmodelCv[i]->SetSelectedVar(m_SelectedVar,0);
      }

      if (currentSelectedVarNb > (int) m_CriterionBestValues.size())
      {
        m_CriterionBestValues.push_back(criterionVal[argMaxValue]);
        m_BestSets.push_back(m_SelectedVar);
        bestSetsPools.push_back(variablesPool);
      }
      else
      {
        m_CriterionBestValues[currentSelectedVarNb-1] = criterionVal[argMaxValue];
        m_BestSets[currentSelectedVarNb-1] = m_SelectedVar;
        bestSetsPools[currentSelectedVarNb-1] = variablesPool;
      }

      flagBacktrack = true;

      while (flagBacktrack && (currentSelectedVarNb > 2))
      {

        std::vector<RealType> criterionValBackward(m_SelectedVar.size(),0);

        // Compute criterion function
        if ( (criterion.compare("accuracy") == 0)||(criterion.compare("kappa") == 0)||(criterion.compare("f1mean") == 0) )
        {
          // COULD BE PARALLELIZED but need to see if it is better to parallelize here on each fold or inside ComputeClassifRate
          for (unsigned i = 0; i < m_SubmodelCv.size(); ++i)
            m_SubmodelCv[i]->ComputeClassifRate(criterionValBackward,"backward",m_SelectedVar,criterion);

          // Compute mean instead of keeping sum of criterion for all folds (not necessary)
          for (unsigned i = 0; i < criterionValBackward.size(); ++i)
            criterionValBackward[i] /= m_SubmodelCv.size();
        }
        else if (criterion.compare("jm") == 0)
          ComputeJM(criterionValBackward,"backward",m_SelectedVar);
        else if (criterion.compare("divkl") == 0)
          ComputeDivKL(criterionValBackward,"backward",m_SelectedVar);

        argMaxValue = std::distance(criterionValBackward.begin(), std::max_element(criterionValBackward.begin(), criterionValBackward.end()));

        if (criterionValBackward[argMaxValue] > m_CriterionBestValues[currentSelectedVarNb-2])
        {
          currentSelectedVarNb--;

          variablesPool.push_back(m_SelectedVar[argMaxValue]);
          m_SelectedVar.erase(m_SelectedVar.begin()+argMaxValue);

          m_CriterionBestValues[currentSelectedVarNb-1] = criterionValBackward[argMaxValue];
          m_BestSets[currentSelectedVarNb-1] = m_SelectedVar;
          bestSetsPools[currentSelectedVarNb-1] = variablesPool;
        }
        else
          flagBacktrack = false;
      }
    }
  }
}

/***************************************/
/******     Criterion functions    *****/
/***************************************/

/** Compute criterion for overall accuracy, Cohen's kappa and F1-mean */
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

    // COULD BE PARALLELIZED
    for (unsigned int k = 0; k < variablesPool.size(); ++k)
    {
      typename TargetListSampleType::Pointer TargetListSample    = TargetListSampleType::New();
      typename TargetListSampleType::Pointer RefTargetListSample = TargetListSampleType::New();
      typename ConfusionMatrixType::Pointer confM                = ConfusionMatrixType::New();

      // Predict labels with variables k added
      for (unsigned int i = 0; i < m_Fold.size(); ++i)
      {
        for (unsigned j = 0; j < Superclass::m_NbSpl[i]; ++j)
        {
          sample = m_Fold[i]->GetMeasurementVectorByIndex(j);

          for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
            scores[c] = (sample[k] - Superclass::m_Means[c][k])*(sample[k] - Superclass::m_Means[c][k]) / Superclass::m_Covariances[c](k,k) + log(Superclass::m_Covariances[c](k,k)) - m_Logprop[c];

          res[0] = Superclass::m_MapOfIndices.at(std::distance(scores.begin(), std::min_element(scores.begin(), scores.end())));
          TargetListSample->PushBack(res);
          res[0] = Superclass::m_MapOfIndices.at(i);
          RefTargetListSample->PushBack(res);
        }
      }

      // Evaluate classification rate
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
      else if (criterion.compare("f1mean") == 0)
      {
        typename ConfusionMatrixType::MeasurementType Fscores = confM->GetFScores();
        RealType meanFscores = 0;
        for (unsigned i = 0; i < Fscores.Size(); ++i)
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
    std::vector<MatrixType> lambdaQ(Superclass::m_ClassNb,MatrixType(selectedVarNb,selectedVarNb));
    std::vector<MatrixType> subMeans(Superclass::m_ClassNb,MatrixType(selectedVarNb,1));
    VectorType eigenValues(selectedVarNb);
    std::vector<RealType> logdet(Superclass::m_ClassNb);

    // Compute inv of covariance matrix and logdet
    for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
    {
      ExtractVectorToColMatrix(m_SelectedVar, Superclass::m_Means[c], subMeans[c]);
      ExtractSubSymmetricMatrix(m_SelectedVar,Superclass::m_Covariances[c],subCovariances);
      Superclass::Decomposition(subCovariances, Q, eigenValues);

      lambdaQ[c]= vnl_diag_matrix<RealType>(eigenValues.apply(sqrt)).invert_in_place() * Q.transpose();
      invCov[c] = Q * (vnl_diag_matrix<RealType>(eigenValues).invert_in_place() * Q.transpose());
      logdet[c] = eigenValues.apply(log).sum();
    }

    InputSampleType sample;
    TargetSampleType res;
    std::vector<RealType> scores(Superclass::m_ClassNb);
    std::vector<RealType> alpha(Superclass::m_ClassNb);
    std::vector<RealType> logdet_update(Superclass::m_ClassNb);
    std::vector<MatrixType> v(Superclass::m_ClassNb,MatrixType(1,selectedVarNb));
    MatrixType u(selectedVarNb,1);
    // VectorType input(Superclass::m_FeatNb);
    MatrixType subInput(selectedVarNb,1);

    // Intermediate variables to reduce allocation/free time
    MatrixType subInput_c(selectedVarNb,1);
    RealType quadraticTermUpdate;

    // COULD BE PARALLELIZED
    for (unsigned int k = 0; k < variablesPool.size(); ++k)
    {
      typename TargetListSampleType::Pointer TargetListSample    = TargetListSampleType::New();
      typename TargetListSampleType::Pointer RefTargetListSample = TargetListSampleType::New();
      typename ConfusionMatrixType::Pointer confM = ConfusionMatrixType::New();

      // Predict labels with variables k added
      if (direction.compare("forward")==0)
      {
        for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
        {
          ExtractReducedColumn(variablesPool[k],m_SelectedVar,Superclass::m_Covariances[c],u);
          alpha[c] = Superclass::m_Covariances[c](variablesPool[k],variablesPool[k]) - (u.transpose() * invCov[c] *u)(0,0);
          if (alpha[c] < std::numeric_limits<RealType>::epsilon())
            alpha[c] = std::numeric_limits<RealType>::epsilon();

          logdet_update[c] = log(alpha[c]) + logdet[c];
          v[c] = -1/alpha[c] * (invCov[c]*u).transpose();
        }

        for (unsigned int i = 0; i < m_Fold.size(); ++i)
        {
          for (unsigned j = 0; j < Superclass::m_NbSpl[i]; ++j)
          {
            sample = m_Fold[i]->GetMeasurementVectorByIndex(j);

            // Convert input data
            for (int n = 0; n < selectedVarNb; ++n)
              subInput(n,0) = sample[m_SelectedVar[n]];

            for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
            {
              subInput_c = subInput - subMeans[c];
              quadraticTermUpdate = (v[c]*subInput_c)(0,0) + 1/alpha[c] * (sample[variablesPool[k]] - Superclass::m_Means[c][variablesPool[k]]);
              scores[c] =  pow((lambdaQ[c]*subInput_c).fro_norm(),2) + alpha[c]*quadraticTermUpdate*quadraticTermUpdate + logdet_update[c] - m_Logprop[c];
            }

            res[0] = Superclass::m_MapOfIndices.at(std::distance(scores.begin(), std::min_element(scores.begin(), scores.end())));
            TargetListSample->PushBack(res);
            res[0] = Superclass::m_MapOfIndices.at(i);
            RefTargetListSample->PushBack(res);
          }
        }
      }
      else if (direction.compare("backward")==0)
      {
        for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
        {
          alpha[c] = 1/invCov[c](k,k);
          if (alpha[c] < std::numeric_limits<RealType>::epsilon())
            alpha[c] = std::numeric_limits<RealType>::epsilon();

          logdet_update[c] = logdet[c] - log(alpha[c]);
          v[c] = invCov[c].get_n_rows(k,1);
        }

        for (unsigned int i = 0; i < m_Fold.size(); ++i)
        {
          for (unsigned j = 0; j < Superclass::m_NbSpl[i]; ++j)
          {
            sample = m_Fold[i]->GetMeasurementVectorByIndex(j);

            // Convert input data
            for (int n = 0; n < selectedVarNb; ++n)
              subInput(n,0) = sample[m_SelectedVar[n]];

            for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
            {
              subInput_c = subInput - subMeans[c];
              quadraticTermUpdate = (v[c]*subInput_c)(0,0);
              scores[c] =  (subInput_c.transpose()*invCov[c]*subInput_c)(0,0)  - alpha[c]*quadraticTermUpdate*quadraticTermUpdate + logdet_update[c] - m_Logprop[c];
            }

            res[0] = Superclass::m_MapOfIndices.at(std::distance(scores.begin(), std::min_element(scores.begin(), scores.end())));
            TargetListSample->PushBack(res);
            res[0] = Superclass::m_MapOfIndices.at(i);
            RefTargetListSample->PushBack(res);
          }
        }
      }

      // Evaluate classification rate
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
      else if (criterion.compare("f1mean") == 0)
      {
        typename ConfusionMatrixType::MeasurementType Fscores = confM->GetFScores();
        RealType meanFscores = 0;
        for (unsigned i = 0; i < Fscores.Size(); ++i)
          meanFscores += (RealType) Fscores[i];
        criterionVal[k] += meanFscores/Superclass::m_ClassNb;
      }
    }
  }
}

/** Compute criterion for Jeffrey-Matusita distance */
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
    // Precompute 0.5*logdet(cov)
    for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
      for (unsigned int j = 0; j < variablesPool.size(); ++j)
        halfedLogdet[c][j] = 0.5*log(Superclass::m_Covariances[c](j,j));

    RealType md, cs, bij;

    // Compute J-M distance
    for (unsigned int c1 = 0; c1 < Superclass::m_ClassNb; ++c1)
    {
      for (unsigned int c2 = c1+1; c2 < Superclass::m_ClassNb; ++c2)
      {
        std::vector<int>::iterator varIt = variablesPool.begin();
        for (unsigned int j = 0; j < variablesPool.size(); ++j)
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
    RealType logdet, alpha=0;
    MatrixType u(selectedVarNb,1);

    // Precompute 0.5*logdet(cov)
    std::vector<int>::iterator varIt;
    // COULD BE PARALLELIZED
    for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
    {
      ExtractSubSymmetricMatrix(m_SelectedVar,Superclass::m_Covariances[c],subCovariances[c]);
      Superclass::Decomposition(subCovariances[c], Q, eigenValues);

      invCov = Q * (vnl_diag_matrix<RealType>(eigenValues).invert_in_place() * Q.transpose());
      logdet = eigenValues.apply(log).sum();

      varIt = variablesPool.begin();
      for (unsigned int j = 0; j < variablesPool.size(); ++j)
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
    RealType logdet_c1c2=0, cst_feat=0, bij;
    MatrixType md(selectedVarNb,1);
    MatrixType extractUTmp(selectedVarNb,1);

    // Extract means
    std::vector<MatrixType> subMeans(Superclass::m_ClassNb,MatrixType(selectedVarNb,1));
    for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
      ExtractVectorToColMatrix(m_SelectedVar, Superclass::m_Means[c], subMeans[c]);

    // Compute JM
    for (unsigned int c1 = 0; c1 < Superclass::m_ClassNb; ++c1)
    {
      for (unsigned int c2 = c1+1; c2 < Superclass::m_ClassNb; ++c2)
      {
        cs = 0.5*(subCovariances[c1] + subCovariances[c2]);
        Superclass::Decomposition(cs, Q, eigenValues);

        invCov = Q * (vnl_diag_matrix<RealType>(eigenValues).invert_in_place() * Q.transpose());
        logdet = eigenValues.apply(log).sum();

        varIt = variablesPool.begin();
        // COULD BE PARALLELIZED
        for (unsigned int k = 0; k < variablesPool.size(); ++k)
        {
          if (direction.compare("forward")==0)
          {
            md = subMeans[c1] - subMeans[c2];

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

/** Compute criterion for Kullback–Leibler divergence */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::ComputeDivKL(std::vector<RealType> & divKL, const std::string direction, std::vector<int> & variablesPool)
{
  // Get info
  int selectedVarNb = (int) m_SelectedVar.size();

  if (m_SelectedVar.empty())
  {
    RealType alpha1,alpha2;

    // Compute KL divergence
    // COULD BE PARALLELIZED
    for (unsigned int k = 0; k < variablesPool.size(); ++k)
    {
      for (unsigned int c1 = 0; c1 < Superclass::m_ClassNb; ++c1)
      {
        alpha1 = 1/Superclass::m_Covariances[c1](variablesPool[k],variablesPool[k]);
        if (alpha1 < std::numeric_limits<RealType>::epsilon())
          alpha1 = std::numeric_limits<RealType>::epsilon();

        for (unsigned int c2 = c1+1; c2 < Superclass::m_ClassNb; ++c2)
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
    int newVarNb=0;

    // Define nb of variables
    if (direction.compare("forward")==0)
      newVarNb = selectedVarNb + 1;
    else if (direction.compare("backward")==0)
      newVarNb = selectedVarNb - 1;
    std::vector<MatrixType> invCov_update(Superclass::m_ClassNb,MatrixType(newVarNb,newVarNb));

    // Extract and decompose cov matrices
    // COULD BE PARALLELIZED
    for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
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
    // COULD BE PARALLELIZED
    for (unsigned int k = 0; k < variablesPool.size(); ++k)
    {
      // commpute update cst
      if (direction.compare("forward")==0)
      {
        std::vector<int>::iterator varIt = m_SelectedVar.begin();
        for (int i = 0; i < selectedVarNb; ++i)
        {
          newSelectedVar[i] = *varIt;
          varIt++;
        }
        newSelectedVar[newVarNb-1] = variablesPool[k];

        for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
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
        for (unsigned i = 0; i < (unsigned) selectedVarNb; ++i)
        {
          if (i!=k)
          {
            *varIt = m_SelectedVar[i];
            varIt++;
          }
        }

        for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
        {
          ExtractSubSymmetricMatrix(newSelectedVar,Superclass::m_Covariances[c],subMatrix);
          ExtractReducedColumn(variablesPool[k],newSelectedVar,Superclass::m_Covariances[c],u);
          invCov_update[c] = subMatrix - 1/invCov[c](k,k) * u * u.transpose();
        }
      }

      // Extract means
      std::vector<MatrixType> subCovariances(Superclass::m_ClassNb,MatrixType(newVarNb,newVarNb));
      for (unsigned int c = 0; c < Superclass::m_ClassNb; ++c)
      {
        ExtractVectorToColMatrix(newSelectedVar, Superclass::m_Means[c], subMeans[c]);
        ExtractSubSymmetricMatrix(newSelectedVar,Superclass::m_Covariances[c],subCovariances[c]);
      }

      // Compute KL divergence
      for (unsigned int c1 = 0; c1 < Superclass::m_ClassNb; ++c1)
      {
        for (unsigned int c2 = c1+1; c2 < Superclass::m_ClassNb; ++c2)
        {
          md       = subMeans[c1] - subMeans[c2];
          divKL[k] += Superclass::m_Proportion[c1] * Superclass::m_Proportion[c2] * 0.5 * ( vnl_trace(invCov_update[c2]*subCovariances[c1] + invCov_update[c1]*subCovariances[c2]) + (md.transpose()*(invCov_update[c1]+invCov_update[c2])*md)(0,0) );
        }
      }
    }
  }
}

/***************************************/
/********   Classification    **********/
/***************************************/

/** Train the machine learning model */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Train()
{
  Superclass::Train();
}

/** Predict values using the model */
template <class TInputValue, class TTargetValue>
typename GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::TargetSampleType
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Predict(const InputSampleType & rawInput, ConfidenceValueType *quality) const
{
  if (m_SelectedVar.empty())
  {
    return Superclass::Predict(rawInput, quality);
  }
  else
  {
    // Convert input data
    VectorType subInput(m_VarNbPrediction);
    for (int i = 0; i < m_VarNbPrediction; ++i)
      subInput[i] = rawInput[m_SelectedVar[i]];

    // Compute decision function
    std::vector<RealType> decisionFct(Superclass::m_CstDecision);
    VectorType lambdaQInputC(m_VarNbPrediction);
    VectorType input_c(m_VarNbPrediction);
    for (unsigned int i = 0; i < Superclass::m_ClassNb; ++i)
    {
      input_c = subInput - m_SubMeans[i];
      lambdaQInputC = Superclass::m_LambdaQ[i] * input_c;

      // Add sum of squared elements
      decisionFct[i] += lambdaQInputC.squared_magnitude();
    }

    int argmin = std::distance(decisionFct.begin(), std::min_element(decisionFct.begin(), decisionFct.end()));

    TargetSampleType res;
    res[0] = Superclass::m_MapOfIndices.at(argmin);

    // Compute confidence (optional)
    if (quality != NULL)
    {
      RealType prob = 0;
      for (unsigned int i = 0; i < decisionFct.size(); ++i)
        prob += exp(-0.5*(decisionFct[i]-decisionFct[argmin]));
      *quality = (ConfidenceValueType) ( 1 / prob);
    }

    return res;
  }
}

/***************************************/
/*********     Read/Write    ***********/
/***************************************/

/** Save the model to file */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Save(const std::string & filename, const std::string & name)
{
  // Save GMM model
  Superclass::Save(filename,name);

  // Save selection model
  if (m_SelectedVar.size() != 0)
  {
    std::string suffix = "_selection";
    std::string selectionFilename = filename;
    std::size_t extPosition = selectionFilename.rfind(".");
    if (extPosition > selectionFilename.length())
      extPosition = selectionFilename.length();
      
    selectionFilename.insert(extPosition,suffix);
    std::ofstream ofs(selectionFilename.c_str(), std::ios::out);

    // Store number of selected variables
    ofs << "nbSelectedVar: " << m_SelectedVar.size() << std::endl;

    // Store number of selected features to use
    ofs << "nbSelectedVarToUse: " << m_VarNbPrediction << std::endl;

    // Store vector of selected features
    ofs << "bestSelectedVarSetsForEachNbOfSelectedVar:" << std::endl;
    for (unsigned int i = 0; i < m_BestSets.size(); ++i)
    {
      for (unsigned int j = 0; j < m_BestSets[i].size(); ++j)
        ofs << m_BestSets[i][j] << " ";
      ofs << std::endl;
    }

    // Set writing precision (need c++11 to not hardcode value of double precision)
    // ofs.precision(std::numeric_limits<double>::max_digits10);
    ofs.precision(17);

    // Store vector of criterion functions values with the corresponding number of features used
    ofs << "criterionfctEvolution:" << std::endl;
    for (unsigned int i = 0; i < m_SelectedVar.size(); ++i)
      ofs << i+1 << " " << m_CriterionBestValues[i] << std::endl;

    ofs.close();
  }
}

/** Load the model from file */
template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::Load(const std::string & filename, const std::string & name)
{
  Superclass::Load(filename,name);

  std::string suffix = "_selection";
  std::string selectionFilename = filename;
  std::size_t extPosition = selectionFilename.rfind(".");
  selectionFilename.insert(extPosition,suffix);
  std::ifstream ifs(selectionFilename.c_str(), std::ios::in);

  if(!ifs)
  {
    std::cerr<<"Could not found/read file "<<selectionFilename<<std::endl;
  }
  else
  {
    int selectedVarNb;
    std::string dump;

    ifs >> dump;
    ifs >> selectedVarNb;

    // Load number of selected features to use
    ifs >> dump;
    ifs >> m_VarNbPrediction;

    // Allocation
    m_BestSets.resize(selectedVarNb);
    m_CriterionBestValues.resize(selectedVarNb);

    // Load selected variables
    ifs >> dump;
    for (int i = 0; i < selectedVarNb; ++i)
    {
      m_BestSets[i].resize(i+1);
      for (int j = 0; j < i+1; ++j)
        ifs >> m_BestSets[i][j];
    }

    // Load criterion function values
    ifs >> dump;
    for (int i = 0; i < selectedVarNb; ++i)
    {
      ifs >> dump;
      ifs >> m_CriterionBestValues[i];
    }

    // Variables to use for prediction
    SetSelectedVar(m_BestSets[m_VarNbPrediction-1]);
  }

  ifs.close();
}

template <class TInputValue, class TTargetValue>
void
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  // Call superclass implementation
  Superclass::PrintSelf(os,indent);
}

/***************************************/
/********       Accessors       ********/
/***************************************/

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
  m_VarNbPrediction = varSubSet.size();
  std::vector<int>().swap(m_SelectedVar);
  m_SelectedVar.resize(m_VarNbPrediction);
  for (int i = 0; i < m_VarNbPrediction; ++i)
    m_SelectedVar[i] = varSubSet[i];

  if (recompute == 1)
  {
    // Precomputation of terms use for prediction //
    Superclass::m_Q.clear();
    Superclass::m_LambdaQ.clear();
    Superclass::m_EigenValues.clear();
    Superclass::m_CstDecision.clear();
    m_SubMeans.clear();

    Superclass::m_Q.resize(Superclass::m_ClassNb,MatrixType(m_VarNbPrediction,m_VarNbPrediction));
    Superclass::m_EigenValues.resize(Superclass::m_ClassNb,VectorType(m_VarNbPrediction));
    MatrixType subCovariance(m_VarNbPrediction,m_VarNbPrediction);
    Superclass::m_CstDecision.resize(Superclass::m_ClassNb,0);
    Superclass::m_LambdaQ.resize(Superclass::m_ClassNb, MatrixType(m_VarNbPrediction,m_VarNbPrediction));
    m_SubMeans.resize(Superclass::m_ClassNb, VectorType(m_VarNbPrediction));

    RealType lambda;
    for ( unsigned int i = 0; i < Superclass::m_ClassNb; ++i )
    {
      // Decompose covariance matrix in eigenvalues/eigenvectors
      ExtractSubSymmetricMatrix(m_SelectedVar,Superclass::m_Covariances[i],subCovariance);
      Superclass::Decomposition(subCovariance, Superclass::m_Q[i], Superclass::m_EigenValues[i]);

      // Extract mean corresponding to selected variables
      ExtractVector(m_SelectedVar,Superclass::m_Means[i],m_SubMeans[i]);

      // Precompute lambda^(-1/2) * Q and log(det lambda)
      for (int j = 0; j < m_VarNbPrediction; ++j)
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
::SetVarNbPrediction(int varNb)
{
  if ((unsigned)varNb <= m_BestSets.size())
  {
    SetSelectedVar(m_BestSets[varNb-1]);
  }
  else
  {
    std::cout << "Warning: Attempt to use more features than the number features identified during selection.\n Number of features used for prediction is set to max (" << m_BestSets.size() << ")" << std::endl;
    SetSelectedVar(m_BestSets[m_BestSets.size()-1]);
  }
}

template <class TInputValue, class TTargetValue>
std::vector<double>
GMMSelectionMachineLearningModel<TInputValue,TTargetValue>
::GetCriterionBestValues()
{
  return m_CriterionBestValues;
}

} //end namespace otb

#endif
