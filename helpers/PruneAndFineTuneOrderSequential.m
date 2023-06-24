function [numParameterRemaining,costs,crossEnt,accuracy] = PruneAndFineTuneOrderSequential(W,B,parOrder,numParPruneStep,epochs,batchSize,eta,decayRate,dataTrain,targetTrain,dataTest,targetTest)
  % prunes numParPruneStep parameters in each iteration until fewer
  % then numParPruneStep parameters remain
  % the order in which the parameters are removed is given by parOrder, 
  % containing a list of indices of all parameters, sorted from 
  % most important to least important

  [n,nIn] = size(W{1});
  d = numel(W);
  [nOut,~] = size(W{d});
  numParameter = length(LinearizeParameter(W,B));
  numIterations = floor(numParameter/numParPruneStep);
  numParameterRemaining = zeros(numIterations,1);
  costs = zeros(numIterations,2);
  crossEnt = zeros(numIterations,2);
  accuracy = zeros(numIterations,2);
  

  for ii=1:numIterations

    % get linearized representation of all weights
    p = LinearizeParameter(W,B);

    % prune according to indices given by parOrder
    pIndex = parOrder(end-ii*numParPruneStep+1:end);
    p(pIndex) = 0;

    % reconstruct  and fine tune
    [W,B,indW,indB] = UnlinearizeParameter(p,d,n,nIn,nOut);
    [W,B] = MyGradientDescend(W,B,indW,indB,dataTrain,targetTrain,epochs,batchSize,eta,decayRate,'True');   

    % compute various metrics
    costs(ii,1) = CostOnData(W,B,dataTrain,targetTrain);
    costs(ii,2) = CostOnData(W,B,dataTest,targetTest);
    crossEnt(ii,1) = CrossEntropy(W,B,dataTrain,targetTrain);
    crossEnt(ii,2) = CrossEntropy(W,B,dataTest,targetTest);
    accuracy(ii,1) = Accuracy(W,B,dataTrain,targetTrain);
    accuracy(ii,2) = Accuracy(W,B,dataTest,targetTest);
    numParameterRemaining(ii) = nnzCell(indW,indB);

  end
end