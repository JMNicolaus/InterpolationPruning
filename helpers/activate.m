function [y,dy] = activate(x)
% returns logistic sigmoid activiation and the derivative
  y  = 1./(1+exp(-1*x));
  dy = y.*(1-y);
end