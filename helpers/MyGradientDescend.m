function [W,b,gradW1,gradB1] = MyGradientDescend_3(W,b,indW,indB,data,target,nEpochs,batchSize,eta,decayRate,ReturnFullGrad)
%MYGRADIENTDESCEND trains the NN given by W,B for the data and target
% nEpochs is the number of gradient steps
% batchSize is the number of training data in each step
% if batchSize == number of datapoints this is the ordinary gradient
% descent method
% indW and indB specify which parameters should be learned
% supply {} in place of indW and indB to calculate the gradient step for
% all parameters, i.e. ignoring ''important'' parameters
% FullGrad is string 'true' or 'false'


    %assert(nargin>=7, 'Please specify W,B,indW,indB,data and target');
    %assert(nargin<=9,'Too many input Arguments');

    d = numel(b);
    z = cell(1,d);
    a = cell(1,d);
    D = cell(1,d);
    del = cell(1,d);
    % eta = 0.05;    
    [~,N] = size(data);
    flag= ~isempty(indW) && ~isempty(indB); % True if indices are given

    if ~exist('ReturnFullGrad') || isempty(ReturnFullGrad)
      ReturnFullGrad = 'false';
    end
    if ~flag
      ReturnFullGrad = 'true';
    end

    %% Backprop
    for ep = 1:nEpochs
      % select batchSize-number of unique datapoints to train this epoch
      indBatch = randperm(N,batchSize);
      dataBatch = data(:,indBatch);
      targetBatch = target(:,indBatch);

      % Calculate (stochastic) gradient
      for k=1:batchSize
          xk = dataBatch(:,k); 
          tk = targetBatch(:,k);        
          for ii = 1:d
              if ii == 1
                  a0 = xk;
                  z{1} = W{1}*a0+b{1};
              else
                  z{ii} = W{ii}*a{ii-1}+b{ii};
              end
              [a{ii},da] = activate(z{ii});
              D{ii} = diag(da);
          end
          del{d}=D{d}*(a{d}-tk);
          for ii=d-1:-1:1
              del{ii} = D{ii}*W{ii+1}'*del{ii+1};
          end 
          for ii = d:-1:1
              if k==1
                  if ii ==1 
                      gradW{ii} = 1/batchSize*del{ii}*a0';
                  else
                      gradW{ii} = 1/batchSize*del{ii}*a{ii-1}';
                  end
                  gradb{ii} = 1/batchSize*del{ii};       
              else
                  if ii ==1 
                      gradW{ii} = gradW{ii} + 1/batchSize*del{ii}*a0';
                  else
                      gradW{ii} = gradW{ii} + 1/batchSize*del{ii}*a{ii-1}';
                  end
                  gradb{ii} = gradb{ii} + 1/batchSize*del{ii};       
              end           
          end
      end

      %% Descend step
      
      if flag
        for ii = d:-1:1
          W{ii}(indW{ii}) = W{ii}(indW{ii}) -eta*gradW{ii}(indW{ii});
          b{ii}(indB{ii}) = b{ii}(indB{ii}) -eta*gradb{ii}(indB{ii});
        end
      else 
         for ii = d:-1:1
          W{ii} = W{ii} -eta*gradW{ii} ;
          b{ii} = b{ii} -eta*gradb{ii} ;
         end  
      end
      eta = eta*decayRate;
    end

    %% Output
    if nargout == 4
      if strcmp(ReturnFullGrad, 'false')
        for ii = 1:d
          gradW1{ii} = gradW{ii}(indW{ii});
          gradB1{ii} = gradb{ii}(indB{ii});
        end
      else
        gradW1 = gradW;
        gradB1 = gradb;
      end
    end
end

