function [W,B,indW1,indB1] = InitializeNetwork(d,n,nIn,nOut,density)
% initialises a network with the given structure

if nargin ==4
  flagSparse = false;
else
  flagSparse = true;
end

W = cell(1,d);
B = cell(1,d);

if ~flagSparse

  W{1} = randn(n,nIn);
  B{1} = randn(n,1);
  for ii = 2:d-1
    W{ii} = randn(n,n);
    B{ii} = randn(n,1);
  end
  W{d} = randn(nOut,n);
  B{d} = randn(nOut,1);
else
  indW = cell(1,d);
  indB = cell(1,d);
  W{1} = sprandn(n,nIn,density);
  B{1} = sprandn(n,1,density);
  indW{1} = W{1} ~=0;
  indB{1} = B{1} ~=0;
  for ii = 2:d-1
    W{ii} = sprandn(n,n,density);
    B{ii} = sprandn(n,1,density);
    indW{ii} = W{ii} ~=0;
    indB{ii} = B{ii} ~=0;
  end
  W{d} = sprandn(nOut,n,density);
  B{d} = sprandn(nOut,1,density);
  indW{d} = W{d} ~=0;
  indB{d} = B{d} ~=0;
end
if nargout == 4
  indW1 = indW;
  indB1 = indB;
end
end

