function [p] = LinearizeParameter(W,B)
%LINEARIZEPARAMETER iterates through the cells W and B and concatenates
% them to form a numParam x 1 vector of the parameters, 
% i.e. [W{1}(:); ...; W{d}(:); B{1}(:); ...;B{d}(:)]
  [~,d] = size(W);
  numParameter = 0;
  for ii=1:d
    numParameter = numParameter + numel(W{ii}) + numel(B{ii});
  end
  p = zeros(numParameter,1);
  ind = 1;
  for ii=1:d
    n = numel(W{ii});
    p(ind:ind+n-1) = W{ii}(:);
    ind = ind+n;
  end
  for ii=1:d
    n = numel(B{ii});
    p(ind:ind+n-1) = B{ii}(:);
    ind = ind+n;
  end
end

