function [cost] = CostOnData(W,B,data,target)
%COSTONDATA an alternative version to CostNet, where the dataset is
%specified as input
  out = EvalNet(data,W,B);
  cost = mean((out-target).^2,'all');
end

