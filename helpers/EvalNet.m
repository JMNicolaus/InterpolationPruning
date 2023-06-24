function [y] = EvalNet(x,W,b)
% W and b are cell arrays
  [~,d] = size(W);
  for ii=1:d
    x = W{ii}*x+b{ii};
    x = activate(x);
  end
  y=x;
end