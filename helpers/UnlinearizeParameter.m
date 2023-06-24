function [W,B,indW,indB] = UnlinearizeParameter(p,d,n,nIn,nOut)
%UNLINEARIZEPARAMETER restores W and B from output of
%LinearizeParameter(W,B)

%% Preallocation
W = cell(1,d);
B = cell(1,d);

if nargout == 4
	indW = cell(1,d);
	indB = cell(1,d);
	W{1} = zeros(n,nIn);
	indW{1} = zeros(n,nIn);
	B{1} = zeros(n,1);
	indB{1} = zeros(n,1);
	for ii= 2:d-1
	  W{ii} = zeros(n);
	  indW{ii} = zeros(n);
	  B{ii} = zeros(n,1);
	  indB{ii} = zeros(n,1);
	end
	W{d} = zeros(nOut,n);
	indW{d} = zeros(nOut,n);
	B{d} = zeros(nOut,1);
	indB{d} = zeros(nOut,1);
else
	W{1} = zeros(n,nIn);
	B{1} = zeros(n,1);
	for ii= 2:d-1
	  W{ii} = zeros(n);
	  B{ii} = zeros(n,1);
	end
	W{d} = zeros(nOut,n);
	B{d} = zeros(nOut,1);
end



%% Reconstruction
ind = 1;
for ii = 1:d
  nEl = numel(W{ii});
  W{ii}(:) = p(ind:ind+nEl-1);
  if nargout == 4
    indW{ii} = W{ii} ~= 0;
  end
  ind = ind+nEl;
end

for ii = 1:d
  nEl = numel(B{ii});
  B{ii}(:) = p(ind:ind+nEl-1);
  if nargout == 4
    indB{ii} = B{ii} ~= 0;
  end
  ind = ind+nEl;
end

