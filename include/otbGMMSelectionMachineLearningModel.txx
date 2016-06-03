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

    for (unsigned int i = 0; i < m_classNb; ++i)
    {
      std::srand ( unsigned ( std::time(0) ) );
      std::vector<InstanceIdentifier> indices;
      for (unsigned j=0; j<m_NbSpl[i]; ++j) indices.push_back(m_classSamples[i]->GetInstanceIdentifier(j));

      std::random_shuffle ( indices.begin(), indices.end() );

      for (int j = 0; j < nfold; ++j)
      {
        submodelCv[j].AddInstanceToFold(indices,j*m_NbSpl[i]/nfold,(j+1)*m_NbSpl[i]/nfold);

        // Update model for each fold
        submodelCv[j].SetMapOfClasses(m_MapOfClasses);
        submodelCv[j].SetMapOfIndices(m_MapOfIndices);
        submodelCv[j].SetClassNb(m_classNb);
        submodelCv[j].SetFeatNb(m_featNb);

        submodelCv[j].AddNbSpl(m_NbSpl[i]/nfold);
        
        submodelCv[j].AddMean(MeanVectorType vector);
        submodelCv[j].AddCovMatrix(MatrixType covMatrix);

      }




// # Update the model for each class
// for c in xrange(self.C):
//     classInd = sp.where(testLabels==(c+1))[0]
//     nk_c     = float(classInd.size)
//     mean_k   = sp.mean(testSamples[classInd,:],axis=0)
//     cov_k    = sp.cov(testSamples[classInd,:],rowvar=0)

//     model_pre_cv[k].nbSpl[c]  = self.nbSpl[c] - nk_c
//     model_pre_cv[k].mean[c,:] = (self.nbSpl[c]*self.mean[c,:]-nk_c*mean_k)/(self.nbSpl[c]-nk_c)
//     model_pre_cv[k].cov[c,:]  = ((self.nbSpl[c]-1)*self.cov[c,:,:] - (nk_c-1)*cov_k - nk_c*self.nbSpl[c]/model_pre_cv[k].nbSpl[c]*sp.outer(self.mean[c,:]-mean_k,self.mean[c,:]-mean_k))/(model_pre_cv[k].nbSpl[c]-1)

//     del classInd,nk_c,mean_k,cov_k

// # Update proportion
// model_pre_cv[k].prop = model_pre_cv[k].nbSpl/(n-nk)

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