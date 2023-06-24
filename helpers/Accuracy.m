function [accuracy] = Accuracy(W,B,data,target)
% computes Top1 accuracy of the network specified by W,b for given 
% data-target pair

[numLabels, numData] = size(target);
out = EvalNet(data,W,B);
prediction = out == repmat(maxk(out,1,1),numLabels,1);
numCorrect = sum(target(prediction));
accuracy = numCorrect/numData;

end