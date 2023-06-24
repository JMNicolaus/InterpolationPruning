function [cMean,cError] = meanCells(c,errorMeasure)
  % computes the mean of the arrays listed in c 
  % over the first dimension of c
  % errorMeasure can be
  % - 'minmax' returns minimum and maximum value over the samples
  % - 'var' returns variance
  % - 'std' returns standard deviation
  % - 'abs' returns abs(max-min)/2
  
  if nargin ==1
    errorMeasure = 'std';
  end
  [n,m] = size(c);
  cMean = cell(1,m);
  if strcmp(errorMeasure,'minmax')
    cError = cell(2,m);
  else
    cError = cell(1,m);
  end

  for jj = 1:m
    [n1,m1]=size(c{1,jj});
    temp = zeros(n1,m1,n);
    % stack entries so we can use predefined functions
    for ii = 1:n
      temp(:,:,ii) = c{ii,jj};
    end
    cMean{jj}=mean(temp,3);

    switch true
      case strcmp(errorMeasure,'minmax')
        cError{1,jj} = min(temp,[],3);
        cError{2,jj} = max(temp,[],3);
      case strcmp(errorMeasure,'var')
        cError{jj} = var(temp,[],3);
      case strcmp(errorMeasure,'std')
        cError{jj} = std(temp,[],3);
      case strcmp(errorMeasure,'abs')
        cError{jj} = abs(max(temp,[],3)-min(temp,[],3))/2;
    end
  end
end