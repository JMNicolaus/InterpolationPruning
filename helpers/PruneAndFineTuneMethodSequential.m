function [numParameterRemaining,costs,crossEnt,accuracy] = PruneAndFineTuneMethodSequential(W,B,method,numParPruneStep,epochs,batchSize,eta,decayRate,dataTrain,targetTrain,dataTest,targetTest)
  % prunes the neural network given by W,B according to method (string)
  % prunes until the number of remaining parameters is less then
  % numParPruneStep.
  % numIterations = floor(numParPrune/numParPruneStep)
 
  % method can be: 
  % - Global Magnitude
  % - Global Gradient Magnitude

  [n,nIn] = size(W{1});
  d = numel(W);
  [nOut,~] = size(W{d});
  numParameter = length(LinearizeParameter(W,B));
  numIterations = floor(numParameter/numParPruneStep);
  numParameterRemaining = zeros(numIterations,1);

  costs = zeros(numIterations,2);
  crossEnt = zeros(numIterations,2);
  accuracy = zeros(numIterations,2);
  indW = cell(size(W));
  indB = cell(size(B));
  
  % compute gradient
  [~,~,gradW,gradB] = MyGradientDescend(W,B,{},{},dataTrain,targetTrain,1,batchSize,eta,decayRate,'True');

  for ii=1:numIterations

    p = LinearizeParameter(W,B);
    pGrad = LinearizeParameter(gradW,gradB);

    switch true
      case strcmp('Global Magnitude',method)
        score = abs(p);
        if ii~=1 
          pIndexPruned = ~logical(LinearizeParameter(indW,indB));
          score(pIndexPruned) = inf;
        end
        [~,pIndex] = mink(score,numParPruneStep);
        p(pIndex) = 0;


      case strcmp('Global Gradient Magnitude',method)
        score = abs(p.*pGrad);
        if ii~=1 
          pIndexPruned = ~logical(LinearizeParameter(indW,indB));
          score(pIndexPruned) = inf;
        end
        [~,pIndex] = mink(score,numParPruneStep);
        p(pIndex) = 0;


      otherwise
        error('Specified method not supported.')


    end

    % reconstruct and fine tune
    [Wpruned,Bpruned,indW,indB] = UnlinearizeParameter(p,d,n,nIn,nOut);
    [W,B,gradW,gradB] = MyGradientDescend(Wpruned,Bpruned,indW,indB,dataTrain,targetTrain,epochs,batchSize,eta,decayRate,'True');

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