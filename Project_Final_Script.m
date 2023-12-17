% Loading the dataset

filename = "C:\Users\anura\OneDrive\Desktop\Machine Learning in Finance\Project\ML_Finance_Project\pivoted.xlsx";
data = readtable(filename);

%%

% Calculating returns for the stocks

returns = diff(data{:, 2:end}) ./ data{1:end-1, 2:end};
returns = [zeros(1, size(returns, 2)); returns];

returnsTable = array2table(returns, 'VariableNames', data.Properties.VariableNames(2:end));

%%

% Finding Correlation between the returns of the stocks

corrMatrix = corr(returnsTable{:,:});

% Set diagonal elements to NaN to ignore them
corrMatrix(eye(size(corrMatrix)) == 1) = NaN;

% Find the maximum correlation value
[maxCorr, ind] = max(corrMatrix(:));

% Find the indices of the maximum value
[row, col] = ind2sub(size(corrMatrix), ind);

% Get the stock names
stockNames = returnsTable.Properties.VariableNames;

% Display the result
fprintf('The highest correlation in returns is %f between %s and %s.\n', maxCorr, stockNames{row}, stockNames{col});

% Plotting the heatmap
figure;
heatmap(stockNames, stockNames, corrMatrix, 'Colormap', jet, 'MissingDataColor', 'white');
title('Heatmap of Stock Return Correlations');

%%

% Considering JPM and BLK based on correlation

corr_stocks = data(:, {'Date', 'JPM', 'BLK'});

% Extract dates and stock prices for JPM and BLK
dates = data.Date;
jpmPrices = data.JPM;
blkPrices = data.BLK;

% Calculate returns for JPM and BLK
jpmReturns = [NaN; diff(jpmPrices) ./ jpmPrices(1:end-1)]; 
blkReturns = [NaN; diff(blkPrices) ./ blkPrices(1:end-1)]; 

% Closing Price Plot
figure;
plot(dates, jpmPrices, 'b-', 'LineWidth', 1.5); 
hold on;
plot(dates, blkPrices, 'r-', 'LineWidth', 1.5); 
hold off;
xlabel('Date');
ylabel('Closing Price');
title('JPM and BLK Closing Prices Over Time');
legend('JPM', 'BLK');
datetick('x', 'yyyy-mm-dd'); 
axis tight;

% Returns Plot
figure;
plot(dates, jpmReturns, 'b-', 'LineWidth', 1.5);
hold on; 
plot(dates, blkReturns, 'r-', 'LineWidth', 1.5); 
hold off; 
xlabel('Date');
ylabel('Returns');
title('JPM and BLK Returns Over Time');
legend('JPM Returns', 'BLK Returns');
datetick('x', 'yyyy-mm-dd'); 
axis tight;

%% Creating Sliding Window

% Extract JPM and BLK stock prices
JPM = data.JPM;
BLK = data.BLK;

% Define the window size M
M = 60; 

% Initialize vectors to store alpha, beta, and residuals
alpha = zeros(length(JPM) - M + 1, 1);
beta = zeros(length(JPM) - M + 1, 1);
residuals = zeros(length(JPM) - M + 1, 1);

%%

% Initialize arrays to store residuals for both models
residualsLinear = zeros(length(JPM) - M + 1, 1);
residualsPoly = zeros(length(JPM) - M + 1, 1);

% Loop over each window
for i = 1:length(JPM) - M + 1
    windowJPM = JPM(i:i+M-1);
    windowBLK = BLK(i:i+M-1);

    % Calculate returns for JPM and BLK
    R_JPM = diff(windowJPM) ./ windowJPM(1:end-1);
    R_BLK = diff(windowBLK) ./ windowBLK(1:end-1);

    % Linear Regression
    X = [ones(length(R_BLK), 1), R_BLK];
    y = R_JPM;
    b = X \ y;
    residualsLinear(i) = R_JPM(end) - (b(2) * R_BLK(end) + b(1));

    % Polynomial Regression
    degree = 3; % Degree of the polynomial
    p = polyfit(R_BLK, R_JPM, degree);
    R_JPM_pred = polyval(p, R_BLK);
    residualsPoly(i) = R_JPM(end) - R_JPM_pred(end);
end

% Calculate cumulative P&L for both models
cumulativePLLinear = cumsum(residualsLinear);
cumulativePLPoly = cumsum(residualsPoly);

% Plotting both cumulative P&Ls
figure;
plot(cumulativePLLinear, 'b-', 'LineWidth', 1.5);
hold on;
plot(cumulativePLPoly, 'r-', 'LineWidth', 1.5);
hold off;
xlabel('Time (Window End Date)');
ylabel('Cumulative P&L');
title('Cumulative P&L Comparison: Linear vs Polynomial Regression');
legend('Linear Regression', 'Polynomial Regression');
%% Polynomial Regression

degree = 2;

% Loop over each window
for i = 1:length(JPM) - M + 1
    
    windowJPM = JPM(i:i+M-1);
    windowBLK = BLK(i:i+M-1);

    % Calculate returns for JPM and BLK
    R_JPM = diff(windowJPM) ./ windowJPM(1:end-1);
    R_BLK = diff(windowBLK) ./ windowBLK(1:end-1);

    % Perform Polynomial regression
    p = polyfit(R_BLK, R_JPM, degree);

    % Predict using the polynomial model
    R_JPM_pred = polyval(p, R_BLK);

    % Calculate and store residuals for the last value in the window
    residuals(i) = R_JPM(end) - R_JPM_pred(end);
end

% Calculate cumulative P&L and reshaping
cumulativePL = cumsum(residuals);
cumulativePL = cumulativePL';  


% Plot the cumulative P&L
figure;
plot(cumulativePL);
xlabel('Time (Window End Date)');
ylabel('Cumulative P&L');
title('Cumulative P&L for JPM and BLK:Polynomial');
%% Linear Regression

% Loop over each window
for i = 1:length(JPM) - M + 1
    
    windowJPM = JPM(i:i+M-1);
    windowBLK = BLK(i:i+M-1);

    % Calculate returns for JPM and BLK
    R_JPM = diff(windowJPM) ./ windowJPM(1:end-1);
    R_BLK = diff(windowBLK) ./ windowBLK(1:end-1);

    % Perform linear regression
    X = [ones(length(R_BLK), 1), R_BLK];
    y = R_JPM; 
    b = X \ y; 

    % Store alpha and beta
    alpha(i) = b(1);
    beta(i) = b(2);

    % Calculate and store residuals
    residuals(i) = R_JPM(end) - (beta(i) * R_BLK(end) + alpha(i));
end

% Calculate cumulative P&L and reshaping
cumulativePL = cumsum(residuals);
cumulativePL = cumulativePL';  


% Plot the cumulative P&L
figure;
plot(cumulativePL);
xlabel('Time (Window End Date)');
ylabel('Cumulative P&L');
title('Cumulative P&L for JPM and BLK Pair Trading');

% Plot alpha over time
figure;
plot(alpha);
xlabel('Time (Window End Date)');
ylabel('Alpha');
title('Alpha Over Time for JPM and BLK Pair Trading');

% Plot beta over time
figure;
plot(beta);
xlabel('Time (Window End Date)');
ylabel('Beta');
title('Beta Over Time for JPM and BLK Pair Trading');

%% Splitting the data and Transforming as per LSTM requirement
   
% Prepare Training and Testing Data
datatrain = cumulativePL(1:344);
datatest = cumulativePL(345:end);

% Reshape Data for LSTM
XTrain = cell(length(datatrain)-1, 1);
YTrain = cell(length(datatrain)-1, 1);

for i = 1:length(datatrain)-1
    XTrain{i} = datatrain(i)'; % Each training feature as a row vector
    YTrain{i} = datatrain(i+1)'; % Each training label as a row vector
end

XTest = cell(length(datatest)-1, 1);
YTest = cell(length(datatest)-1, 1);

for i = 1:length(datatest)-1
    XTest{i} = datatest(i)'; 
    YTest{i} = datatest(i+1)'; 
end

%% Basic LSTM Network

% Define LSTM network architecture
layers = [ ...
    sequenceInputLayer(1)
    lstmLayer(20,'OutputMode','sequence')
    fullyConnectedLayer(1)
    regressionLayer];

% Specify training options
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'MiniBatchSize',1,...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001000, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');


% Train the network
trainedNetwork_1 = trainNetwork(XTrain, YTrain, layers, options);



%%
% Predict on Training Data using the trained network
YPredTrain = predict(trainedNetwork_1, XTrain, 'MiniBatchSize', 1);

% Convert YPredTrain and YTrain from cell arrays to numeric arrays
YPredTrainNumeric = cell2mat(YPredTrain);
YTrainNumeric = cell2mat(YTrain);

% Calculate RMSE for Training Data
rmseTrain = sqrt(mean((YPredTrainNumeric - YTrainNumeric).^2));

% Plotting Predictions vs Actual Data for Training Set
figure;
plot(YTrainNumeric, 'b-', 'LineWidth', 1.5);
hold on;
plot(YPredTrainNumeric, 'r--', 'LineWidth', 1.5);
hold off;
xlabel('Time Steps');
ylabel('Stock Returns');
title(['LSTM Predictions vs Actual Data on Training Set - RMSE: ', num2str(rmseTrain)]);
legend('Actual Data', 'Predicted Data');
%%

% Predict using the trained network
net = predictAndUpdateState(trainedNetwork_1, XTrain);

%%

% Reset network state
net = resetState(net);

% Predict using a single sample
[net, YPred] = predictAndUpdateState(net, YTrain(end));


%%

numTimeStepsTest = numel(XTest);
YPred = cell(numTimeStepsTest, 1); % Initialize YPred as a cell array

for i = 1:numTimeStepsTest
    [net, YPred{i}] = predictAndUpdateState(net, XTest{i}, 'ExecutionEnvironment', 'cpu');
end

% Convert YPred and YTest from cell arrays to numeric arrays
YPredNumeric = cell2mat(YPred);
YTestNumeric = cell2mat(YTest);

% Calculate RMSE
rmse = sqrt(mean((YPredNumeric - YTestNumeric).^2));

%%

% Length of Train and Test
numTimeStepsTrain = numel(datatrain);
numTimeStepsTest = numel(datatest);

% Adjusting the index calculation
idx = (numTimeStepsTrain + 1) : (numTimeStepsTrain + numTimeStepsTest);

figure;
plot(datatrain(1:end-1));
hold on;

% Convert YPred to a numeric array for plotting
YPredNumeric = cell2mat(YPred);

% Adjust the plot command to use YPredNumeric
plot(idx, [cumulativePL(numTimeStepsTrain); YPredNumeric], '.-');

hold off;
xlabel("Days");
ylabel("Returns");
title("Forecast - Polynomial CumulativePL");
legend(["Observed", "Forecast"]);
%% Comparing Forecasting with Testdata

% Initialize YPred as a numeric array
YPred = zeros(size(YTest));

% Reset network state before prediction
net = resetState(trainedNetwork_1);

% Forecasting on Test Data
for i = 1:numel(XTest)
    [net, YPred(i)] = predictAndUpdateState(net, XTest{i}, 'ExecutionEnvironment', 'cpu');
end

% Calculate RMSE
rmse = sqrt(mean((YPred - cell2mat(YTest)).^2));

% Plotting Results
figure;
subplot(2,1,1);
plot(cell2mat(YTest));
hold on;
plot(YPred,'.-');
hold off;
legend(["Actual Test Data", "Forecasted Data"]);
ylabel("Returns");
title("Forecast - Testdata:");

subplot(2,1,2);
stem(YPred - cell2mat(YTest));
xlabel("Days");
ylabel("Error");
title("RMSE = " + rmse);

%% Hyperparameter Tuning

hiddenUnitsOptions = [20, 50, 100]; 
learningRateOptions = [0.01, 0.005, 0.001]; 
epochOptions = [10, 20, 30];  
batchSizeOptions = [1, 2, 8];  

% Initialize variables to store the best hyperparameter configuration
bestRMSE = inf;
bestHiddenUnits = NaN;
bestLearningRate = NaN;
bestEpochs = NaN;
bestBatchSize = NaN;
bestNet = [];

% Loop over all combinations of hyperparameters
for hiddenUnits = hiddenUnitsOptions
    for learningRate = learningRateOptions
        for epochs = epochOptions
            for batchSize = batchSizeOptions

                % Define the LSTM network architecture
                layers = [ ...
                    sequenceInputLayer(1), ...
                    lstmLayer(hiddenUnits, 'OutputMode', 'sequence'), ...
                    fullyConnectedLayer(1), ...
                    regressionLayer];

                % Specify the training options
                options = trainingOptions('adam', ...
                    'MaxEpochs', epochs, ...
                    'MiniBatchSize', batchSize, ...
                    'GradientThreshold', 1, ...
                    'InitialLearnRate', learningRate, ...
                    'LearnRateSchedule', 'piecewise', ...
                    'LearnRateDropPeriod', 125, ...
                    'LearnRateDropFactor', 0.2, ...
                    'Verbose', 0, ...
                    'Plots', 'none'); 
                
                % Train the network
                net = trainNetwork(XTrain, YTrain, layers, options);

                % Predict on the training set and calculate RMSE
                YPredTrain = predict(net, XTrain, 'MiniBatchSize', 1);
                trainRMSE = sqrt(mean((cell2mat(YPredTrain) - cell2mat(YTrain)).^2));

                % If the current RMSE is better than the best recorded RMSE, update the best hyperparameters
                if trainRMSE < bestRMSE
                    bestRMSE = trainRMSE;
                    bestHiddenUnits = hiddenUnits;
                    bestLearningRate = learningRate;
                    bestEpochs = epochs;
                    bestBatchSize = batchSize;
                    bestNet = net;
                end
            end
        end
    end
end

% Display the best hyperparameters
fprintf('Best RMSE on training set: %f\n', bestRMSE);
fprintf('Best hidden units: %d\n', bestHiddenUnits);
fprintf('Best learning rate: %f\n', bestLearningRate);
fprintf('Best epochs: %d\n', bestEpochs);
fprintf('Best mini-batch size: %d\n', bestBatchSize);
%% Plotting for Training Data

% Use the best model to predict on the training set
YPredTrainBest = predict(bestNet, XTrain, 'MiniBatchSize', 1);

% Plot the training set predictions
figure;
subplot(2,1,1);
plot(cell2mat(YTrain));
hold on;
plot(cell2mat(YPredTrainBest), '.-');
hold off;
legend(["Observed", "Predicted"]);
ylabel("Returns");
title("Training Forecast with Best Hyperparameters");

subplot(2,1,2);
stem(cell2mat(YPredTrainBest) - cell2mat(YTrain));
xlabel("Days");
ylabel("Error");
title("Training RMSE with Best Hyperparameters = " + bestRMSE);

%% Testing with Best Model

% Use the best model to predict on the test set
YPredTest = predict(bestNet, XTest, 'MiniBatchSize', 1);
testRMSE = sqrt(mean((cell2mat(YPredTest) - cell2mat(YTest)).^2));

% Display the test RMSE
fprintf('RMSE on test set with best hyperparameters: %f\n', testRMSE);

% Plot the test set predictions
figure;
subplot(2,1,1);
plot(cell2mat(YTest));
hold on;
plot(cell2mat(YPredTest), '.-');
hold off;
legend(["Observed", "Predicted"]);
ylabel("Returns");
title("Test Forecast with Best Hyperparameters");

subplot(2,1,2);
stem(cell2mat(YPredTest) - cell2mat(YTest));
xlabel("Days");
ylabel("Error");
title("Test RMSE with Best Hyperparameters = " + testRMSE);

%% Simple Pair Trading Strategy







