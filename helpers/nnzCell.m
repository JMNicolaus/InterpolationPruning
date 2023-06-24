function x = nnzCell(W,B)
%NNZCELL sums the nuber of nonzero elements in each cell
x =0;
for ii =1:numel(W)
  x =x+ nnz(W{ii})+nnz(B{ii});
end

end

