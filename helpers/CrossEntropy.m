function [crossEntropy] = CrossEntropy(W,B,data,target)
% computes the Cross-Entropy of the network specified by W,b for given 
% data-target pair

out = EvalNet(data,W,B);
sout = softmax(out);
crossEntropy = -sum(target.*log(sout),'all');
end