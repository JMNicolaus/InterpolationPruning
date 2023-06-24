clear all;
clf;
rng(0);
d = 4;  
n = 20; 
nIn = 2;
nOut = 4;
a=-5;
b= 5;
batchSize = 100;
epochs = 1e4;
numParameter = n*nIn + n*nOut + (d-2)*n*n + (d-1)*n +nOut;
decayRate = exp(log(1e-4)/epochs);
[data,target] = GenData4Cat(batchSize,a,b);
[dataTest,targetTest] = GenData4Cat(batchSize,a,b);
eta = 1;


%% collect or load snapshots

% use loadSnaps = true to reproduce the reported graphs .mat files
% use loadSnaps = false to perform new experiments

loadSnaps = true;

if loadSnaps == true
  snapGrads0 = load('./snapGrads0AdditionalTrajectories.mat').snapGrads;
  snapGrads10 = load('./snapGrads10AdditionalTrajectories.mat').snapGrads;
  snapGrads100 = load('./snapGrads100AdditionalTrajectories.mat').snapGrads;
else
  %rng(42)
  numAdditionalSamples = 100;
  numEpochsForAddSamples = 10;
  snap = zeros(numParameter,epochs+numAdditionalSamples*numEpochsForAddSamples);
  snapGrads = zeros(numParameter,epochs+numAdditionalSamples*numEpochsForAddSamples);
  cost = zeros(2,epochs+numAdditionalSamples);
  [Winit,Binit] = InitializeNetwork(d,n,nIn,nOut);
  W = Winit;
  B = Binit;
  
  % collect from training
  etaCurr = eta;
  for ii =1:epochs
    snap(:,ii) = LinearizeParameter(W,B);
    [W,B,gradW,gradB] = MyGradientDescend(W,B,{},{},data,target,1,batchSize,etaCurr,1,'True');
    snapGrads(:,ii) = LinearizeParameter(gradW,gradB);
    cost(1,ii) = CostOnData(W,B,data,target);
    cost(2,ii) = CostOnData(W,B,dataTest,targetTest);
    etaCurr = etaCurr*decayRate;
  end
  
  % collect from random p
  for ii = 1:numAdditionalSamples
    etaCurr = eta;
    for jj=1:numEpochsForAddSamples
      [W,B] = InitializeNetwork(d,n,nIn,nOut);
      [~,~,gradW,gradB] = MyGradientDescend(W,B,{},{},data,target,1,batchSize,etaCurr,1,'True');
      snap(:,epochs+ii*jj) = LinearizeParameter(W,B);
      snapGrads(:,epochs+ii*jj) = LinearizeParameter(gradW,gradB);
      cost(1,epochs+ii*jj) = CostOnData(W,B,data,target);
      cost(2,epochs+ii*jj) = CostOnData(W,B,dataTest,targetTest);
      etaCurr = etaCurr*decayRate;
    end
  end
  snapGrads0 = snapGrads(:,1:epochs);
  snapGrads10 = snapGrads(:,1:epochs+10*numEpochsForAddSamples);
  snapGrads100 = snapGrads;

end
%% SVD
[U0,S0,V0] = svds(snapGrads0,numParameter);
[U10,S10,V10] = svds(snapGrads10,numParameter);
[U100,S100,V100] = svds(snapGrads100,numParameter);

%% plot singular values
%c = linspecer(3,'qualitative');
c = ['#1b9e77';'#d95f02';'#7570b3'];
figure(1)
semilogy(svd(snapGrads0),'color',c(1,:),'LineWidth',1.5)
hold on
plot(svd(snapGrads10),'color',c(2,:),'LineWidth',1.5)
plot(svd(snapGrads100),'color',c(3,:),'LineWidth',1.5)
hold off
grid on
xlabel('i','Interpreter','Latex')
ylabel('singular values of $Y$','Interpreter','Latex')
legend("$n_{a}=0$","$n_{a}=10$","$n_{a}=100$",'Interpreter','Latex')
title('singular values of snapshot matrices $\mathbf{F}_0,\mathbf{F}_{10}$ and $\mathbf{F}_{100}$','Interpreter','Latex')
% savefig(figure(1),"../plots/SingVal2.fig")
% saveas(figure(1),"../plots/SingVal2.eps",'epsc')
%% DEIM
[~,rho0] = DEIM(U0);
[~,rho10] = DEIM(U10);
[~,rho100] = DEIM(U100);


%% Pruning methods
methods = {'DEIM0','DEIM10','DEIM100','Global Magnitude', 'Global Gradient Magnitude','Random'};
numParPruneStep = 10;
numEpochsPruning = 100;
decayRatePruning = exp(log(1e-4)/numEpochsPruning);
etaPruning = 1;
numSamplesPruning=128;

% collect metrics in cell, cell index is given by index of method
costComp = cell(numSamplesPruning,numel(methods));
accComp = cell(numSamplesPruning,numel(methods));
ceComp = cell(numSamplesPruning,numel(methods));


for ii=1:numSamplesPruning

  % initialise networks randomly and pretrain
  rng('shuffle') % make sure to generate different initializations and data
  [data,target] = GenData4Cat(batchSize,a,b);
  [dataTest,targetTest] = GenData4Cat(batchSize,a,b);
  [W,B] = InitializeNetwork(d,n,nIn,nOut);
  [W,B] = MyGradientDescend(W,B,{},{},data,target,epochs,batchSize,eta,decayRate,'True');

  for mIndex = 1:numel(methods)

    switch true

      case strcmp(methods(mIndex),'DEIM0')
        % DEIM pruning
        rng(0)
        [numRem,costComp{ii,mIndex},ceComp{ii,mIndex},accComp{ii,mIndex}] = PruneAndFineTuneOrderSequential(W,B,rho0,...
          numParPruneStep,numEpochsPruning,batchSize,etaPruning,decayRatePruning,...
          data,target,dataTest,targetTest);

      case strcmp(methods(mIndex),'DEIM10')
        % DEIM pruning
        rng(0)
        [numRem,costComp{ii,mIndex},ceComp{ii,mIndex},accComp{ii,mIndex}] = PruneAndFineTuneOrderSequential(W,B,rho10,...
          numParPruneStep,numEpochsPruning,batchSize,etaPruning,decayRatePruning,...
          data,target,dataTest,targetTest);

      case strcmp(methods(mIndex),'DEIM100')
        % DEIM pruning
        rng(0)
        [numRem,costComp{ii,mIndex},ceComp{ii,mIndex},accComp{ii,mIndex}] = PruneAndFineTuneOrderSequential(W,B,rho100,...
          numParPruneStep,numEpochsPruning,batchSize,etaPruning,decayRatePruning,...
          data,target,dataTest,targetTest);

        case strcmp(methods(mIndex),'Random')
        % Random pruning
        rng('shuffle')
        indexSelection = randperm(numParameter)';
        rng(0)
        [numRem,costComp{ii,mIndex},ceComp{ii,mIndex},accComp{ii,mIndex}] = PruneAndFineTuneOrderSequential(W,B,indexSelection,...
          numParPruneStep,numEpochsPruning,batchSize,etaPruning,decayRatePruning,...
          data,target,dataTest,targetTest);

      otherwise
        % pruning with other reference methods
        rng(0)
        [numParRemComp,costComp{ii,mIndex},ceComp{ii,mIndex},accComp{ii,mIndex}] = PruneAndFineTuneMethodSequential(...
          W,B,methods{mIndex},numParPruneStep,...
          numEpochsPruning,batchSize,etaPruning,decayRatePruning,...
          data,target,dataTest,targetTest);
    end
  end
end


%% plot

% set plotting options
errorMeasure = 'std';
plotXScale = 'log';
plotYScale = 'lin';
plotError = true;
plotAsFigure = true;

% set distincitve colors
%c = linspecer(numel(methods),'qualitative');
%c = ['#e41a1c';'#377eb8';'#4daf4a';'#984ea3'];
%c = ['#1b9e77';'#d95f02';'#7570b3';'#e7298a'];
%c = ['#e41a1c';'#377eb8';'#4daf4a';'#984ea3';'#ff7f00';'#ffff33'];
c = ['#1b9e77';'#d95f02';'#7570b3';'#e7298a';'#000000';'#e6ab02'];

% compute mean and error measure of collected samples
[accMean,accMeanError] = meanCells(accComp,errorMeasure);
[ceMean,ceMeanError] = meanCells(ceComp,errorMeasure);
[costMean,costMeanError] = meanCells(costComp,errorMeasure);


% start plotting
figure(2)
clf(2)
plotXVals =[974 494 244 124 64 34 14]
plotYIndices = any(numRem ==[974 494 244 124 64 34 14],2); 
% loop over metrics
for ii = 1:3
  % loop over training and test results
  for jj = 1:2

    switch jj
      case 1
        LabelDataset = "Training Dataset";
      case 2
        LabelDataset = "Test Dataset";
    end

    switch ii
      case 1
        dataPlotM1 = accMean{1,1}(:,jj);
        dataPlotM2 = accMean{1,2}(:,jj);
        dataPlotM3 = accMean{1,3}(:,jj);
        dataPlotM4 = accMean{1,4}(:,jj);
        dataPlotM5 = accMean{1,5}(:,jj);
        dataPlotM6 = accMean{1,6}(:,jj);
        errorM1 = accMeanError{1,1}(:,jj);
        errorM2 = accMeanError{1,2}(:,jj);
        errorM3 = accMeanError{1,3}(:,jj);
        errorM4 = accMeanError{1,4}(:,jj);
        errorM5 = accMeanError{1,5}(:,jj);
        errorM6 = accMeanError{1,6}(:,jj);
        labelX = "$c=\#RemainingParameters / \#TotalParameters$"; %"Number of parameters remaining";
        labelY = "Accuracy";
        labelMethod = "Top1 Accuracy";

      case 2
        dataPlotM1 = ceMean{1,1}(:,jj);
        dataPlotM2 = ceMean{1,2}(:,jj);
        dataPlotM3 = ceMean{1,3}(:,jj);
        dataPlotM4 = ceMean{1,4}(:,jj);
        dataPlotM5 = ceMean{1,5}(:,jj);
        dataPlotM6 = ceMean{1,6}(:,jj);
        errorM1 = ceMeanError{1,1}(:,jj);
        errorM2 = ceMeanError{1,2}(:,jj);
        errorM3 = ceMeanError{1,3}(:,jj);
        errorM4 = ceMeanError{1,4}(:,jj);
        errorM5 = ceMeanError{1,5}(:,jj);
        errorM6 = ceMeanError{1,6}(:,jj);
        labelX = "$c=\#RemainingParameters / \#TotalParameters$"; %"Number of parameters remaining";
        labelY = "Cross Entropy";
        labelMethod = "Cross Entropy";

      case 3
        dataPlotM1 = costMean{1,1}(:,jj);
        dataPlotM2 = costMean{1,2}(:,jj);
        dataPlotM3 = costMean{1,3}(:,jj);
        dataPlotM4 = costMean{1,4}(:,jj);
        dataPlotM5 = costMean{1,5}(:,jj);
        dataPlotM6 = costMean{1,6}(:,jj);
        errorM1 = costMeanError{1,1}(:,jj);
        errorM2 = costMeanError{1,2}(:,jj);
        errorM3 = costMeanError{1,3}(:,jj);
        errorM4 = costMeanError{1,4}(:,jj);
        errorM5 = costMeanError{1,5}(:,jj);
        errorM6 = costMeanError{1,6}(:,jj);
        labelX = "$c=\#RemainingParameters / \#TotalParameters$"; %"Number of parameters remaining";
        labelY = "$L_2$ loss";
        labelMethod = "$L_2$ loss";

    end
    
    if plotAsFigure
      figure((ii-1)*2+jj+1)
    else 
      subplot(3,2,(ii-1)*2+jj)
    end

    if plotError
      h = errorbar(plotXVals,dataPlotM1(plotYIndices),errorM1(plotYIndices),'-o','Color',c(1,:),'LineWidth',1.5);
      hold on
      errorbar(plotXVals,dataPlotM2(plotYIndices),errorM2(plotYIndices),'-.*','Color',c(2,:),'LineWidth',1.5)
      errorbar(plotXVals,dataPlotM3(plotYIndices),errorM3(plotYIndices),'-*','Color',c(3,:),'LineWidth',1.5)
      errorbar(plotXVals,dataPlotM4(plotYIndices),errorM4(plotYIndices),'--*','Color',c(4,:),'LineWidth',1.5)
      errorbar(plotXVals,dataPlotM5(plotYIndices),errorM5(plotYIndices),'-d','Color',c(5,:),'LineWidth',1.5)
      errorbar(plotXVals(1:end-1)',dataPlotM6(plotYIndices(1:end-2),1),errorM6(plotYIndices(1:end-2)),':*','Color',c(6,:),'LineWidth',1.5)
      hold off
    else  
      h = plot(plotXVals,dataPlotM1(plotYIndices),'-o','Color',c(1,:),'LineWidth',1);
      hold on
      plot(plotXVals,dataPlotM2(plotYIndices),'-*','Color',c(2,:),'LineWidth',1)
      plot(plotXVals,dataPlotM3(plotYIndices),'-*','Color',c(3,:),'LineWidth',1)
      plot(plotXVals,dataPlotM4(plotYIndices),'-*','Color',c(4,:),'LineWidth',1)
      plot(plotXVals,dataPlotM5(plotYIndices),'-d','Color',c(5,:),'LineWidth',1)
      plot(plotXVals(1:end-1)',dataPlotM6(plotYIndices(1:end-2),1),'-*','Color',c(6,:),'LineWidth',1)
      hold off
    end

    % set scaling
    set(get(h,'Parent'),'Yscale',plotYScale);
    set(get(h,'Parent'),'Xscale',plotXScale);
    xticks(fliplr(plotXVals))
    xticklabels({'1/64','1/32','1/16','1/8','1/4','1/2','1',})

    % construct title and labels
    title(labelMethod + " on " + LabelDataset,'Interpreter','Latex');
    xlabel(labelX,'Interpreter','Latex');
    ylabel(labelY,'Interpreter','Latex');
    grid on
    if (ii-1)*2+jj == 6
      legend(["$n_{a}=0$","$n_{a}=10$","$n_{a}=100$","Magnitude Pruning (MP)","Gradient Magnitude Pruning (GMP)","Random Pruning (RP)"],'Location','southwest','Interpreter','latex','Box','off')
    end

    % save figure as fig and png
    if plotAsFigure
      fileName = strrep(strrep(strrep(labelY+LabelDataset,' ',''),'$',''),'^','');
%       savefig(figure((ii-1)*2+jj+1),"../plots/Poster/"+fileName+".fig");
%       saveas(figure((ii-1)*2+jj+1),"../plots/Poster/"+fileName+".eps",'epsc');
    end
  end
end


