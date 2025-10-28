%% HW2Q2: ML and MAP Estimators for Cubic Polynomial Regression
% Implementation using the provided data generation function

clear all; close all; clc;

%% Generate Training and Validation Data
Ntrain = 100;
Nvalidate = 1000;

% Use the provided function to generate data
[xTrain, yTrain, xValidate, yValidate] = hw2q2(Ntrain, Nvalidate);

% Ensure y is a column vector
yTrain = yTrain(:);
yValidate = yValidate(:);

fprintf('========================================================================\n');
fprintf('DATA GENERATION COMPLETE\n');
fprintf('========================================================================\n');
fprintf('Training samples: %d\n', Ntrain);
fprintf('Validation samples: %d\n', Nvalidate);
fprintf('Input dimension: %d\n', size(xTrain, 1));
fprintf('\n');

%% Create Cubic Polynomial Feature Matrix
function Phi = create_cubic_features(X)
    % X is 2×N, need to transpose to N×2 for processing
    X = X';
    N = size(X, 1);
    x1 = X(:, 1);
    x2 = X(:, 2);
    
    % Create cubic polynomial features
    % [1, x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3]
    Phi = [ones(N, 1), ...      % w0
           x1, ...               % w1
           x2, ...               % w2
           x1.^2, ...            % w3
           x1.*x2, ...           % w4
           x2.^2, ...            % w5
           x1.^3, ...            % w6
           x1.^2.*x2, ...        % w7
           x1.*x2.^2, ...        % w8
           x2.^3];               % w9
end

%% ML Estimator Implementation
fprintf('========================================================================\n');
fprintf('ML ESTIMATOR\n');
fprintf('========================================================================\n');

% Create design matrix for training data
Phi_train = create_cubic_features(xTrain);

% ML Estimator: w_ML = (Φ'Φ)^(-1)Φ'y
tic;
% Add small regularization for numerical stability
epsilon = 1e-10;
w_ML = (Phi_train' * Phi_train + epsilon * eye(10)) \ (Phi_train' * yTrain);
time_ML = toc;

% Evaluate on training and validation sets
y_pred_train_ML = Phi_train * w_ML;
Phi_val = create_cubic_features(xValidate);
y_pred_val_ML = Phi_val * w_ML;

% Calculate mean squared errors
mse_train_ML = mean((yTrain - y_pred_train_ML).^2);
mse_val_ML = mean((yValidate - y_pred_val_ML).^2);

fprintf('ML Estimator Results:\n');
fprintf('  Training time: %.4f seconds\n', time_ML);
fprintf('  Training MSE: %.6f\n', mse_train_ML);
fprintf('  Validation MSE: %.6f\n', mse_val_ML);
fprintf('  ||w_ML||₂ = %.4f\n', norm(w_ML));
fprintf('  Condition number of Φ''Φ: %.2e\n', cond(Phi_train' * Phi_train));
fprintf('\n');

%% MAP Estimator Implementation for Various γ Values
fprintf('========================================================================\n');
fprintf('MAP ESTIMATOR WITH VARYING γ\n');
fprintf('========================================================================\n');

% Estimate noise variance from ML residuals
sigma2 = mse_train_ML;
fprintf('Estimated noise variance σ² = %.6f\n\n', sigma2);

% Define range of gamma values (10^-5 to 10^5)
gamma_powers = -5:0.5:5;
gamma_values = 10.^gamma_powers;
n_gammas = length(gamma_values);

% Initialize storage
w_MAP_all = zeros(10, n_gammas);
mse_train_MAP = zeros(n_gammas, 1);
mse_val_MAP = zeros(n_gammas, 1);

% Train MAP estimators for each gamma
fprintf('Training MAP estimators for different γ values:\n');
fprintf('------------------------------------------------\n');

for i = 1:n_gammas
    gamma = gamma_values(i);
    
    % MAP Estimator: w_MAP = (Φ'Φ + λI)^(-1)Φ'y, where λ = σ²/γ
    lambda = sigma2 / gamma;
    w_MAP = (Phi_train' * Phi_train + lambda * eye(10)) \ (Phi_train' * yTrain);
    w_MAP_all(:, i) = w_MAP;
    
    % Evaluate
    y_pred_train = Phi_train * w_MAP;
    y_pred_val = Phi_val * w_MAP;
    
    mse_train_MAP(i) = mean((yTrain - y_pred_train).^2);
    mse_val_MAP(i) = mean((yValidate - y_pred_val).^2);
    
    % Display progress for selected values
    if mod(i, 4) == 1 || i == n_gammas
        fprintf('γ = %.2e: Train MSE = %.6f, Val MSE = %.6f, ||w|| = %.4f\n', ...
                gamma, mse_train_MAP(i), mse_val_MAP(i), norm(w_MAP));
    end
end

% Find optimal gamma (minimum validation MSE)
[min_mse_val, opt_idx] = min(mse_val_MAP);
gamma_optimal = gamma_values(opt_idx);
w_MAP_optimal = w_MAP_all(:, opt_idx);

fprintf('\n');
fprintf('========================================================================\n');
fprintf('OPTIMAL MAP ESTIMATOR\n');
fprintf('========================================================================\n');
fprintf('Optimal γ = %.4e\n', gamma_optimal);
fprintf('Optimal λ = σ²/γ = %.4e\n', sigma2/gamma_optimal);
fprintf('Validation MSE at optimal γ: %.6f\n', min_mse_val);
fprintf('Training MSE at optimal γ: %.6f\n', mse_train_MAP(opt_idx));
fprintf('||w_MAP_optimal||₂ = %.4f\n', norm(w_MAP_optimal));
fprintf('\nImprovement over ML:\n');
fprintf('  MSE reduction: %.2f%%\n', 100*(mse_val_ML - min_mse_val)/mse_val_ML);
fprintf('  ||w_MAP - w_ML||₂ = %.4f\n', norm(w_MAP_optimal - w_ML));

%% Visualization
figure(3);
subplot(2, 3, 1);
semilogx(gamma_values, mse_train_MAP, 'b-', 'LineWidth', 2);
hold on;
semilogx(gamma_values, mse_val_MAP, 'r-', 'LineWidth', 2);
semilogx([min(gamma_values), max(gamma_values)], [mse_val_ML, mse_val_ML], 'k--', 'LineWidth', 1.5);
semilogx(gamma_optimal, min_mse_val, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
xlabel('Prior Variance γ', 'FontWeight', 'bold');
ylabel('Mean Squared Error', 'FontWeight', 'bold');
title('MSE vs Prior Variance γ', 'FontWeight', 'bold');
legend('Training', 'Validation', 'ML Baseline', 'Optimal γ', 'Location', 'best');
grid on;

subplot(2, 3, 2);
param_norms = sqrt(sum(w_MAP_all.^2, 1));
loglog(gamma_values, param_norms, 'b-', 'LineWidth', 2);
hold on;
loglog([min(gamma_values), max(gamma_values)], [norm(w_ML), norm(w_ML)], 'k--', 'LineWidth', 1.5);
loglog(gamma_optimal, norm(w_MAP_optimal), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
xlabel('Prior Variance γ', 'FontWeight', 'bold');
ylabel('||w||₂', 'FontWeight', 'bold');
title('Parameter Norm vs γ', 'FontWeight', 'bold');
legend('MAP', 'ML', 'Optimal γ', 'Location', 'best');
grid on;

subplot(2, 3, 3);
distances_to_ML = zeros(n_gammas, 1);
for i = 1:n_gammas
    distances_to_ML(i) = norm(w_MAP_all(:, i) - w_ML);
end
loglog(gamma_values, distances_to_ML, 'r-', 'LineWidth', 2);
hold on;
loglog(gamma_optimal, distances_to_ML(opt_idx), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
xlabel('Prior Variance γ', 'FontWeight', 'bold');
ylabel('||w_{MAP} - w_{ML}||₂', 'FontWeight', 'bold');
title('Distance from MAP to ML', 'FontWeight', 'bold');
grid on;

subplot(2, 3, 4);
% Parameter evolution for first 4 parameters
colors = lines(4);
for j = 1:4
    semilogx(gamma_values, w_MAP_all(j, :), 'LineWidth', 1.5, 'Color', colors(j, :));
    hold on;
end
for j = 1:4
    semilogx([min(gamma_values), max(gamma_values)], [w_ML(j), w_ML(j)], ...
             '--', 'LineWidth', 1, 'Color', colors(j, :));
end
xlabel('Prior Variance γ', 'FontWeight', 'bold');
ylabel('Parameter Value', 'FontWeight', 'bold');
title('Parameter Evolution with γ', 'FontWeight', 'bold');
legend('w₀', 'w₁', 'w₂', 'w₃', 'Location', 'best');
grid on;

subplot(2, 3, 5);
scatter(yValidate, y_pred_val_ML, 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
y_pred_val_MAP_opt = Phi_val * w_MAP_optimal;
scatter(yValidate, y_pred_val_MAP_opt, 10, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
plot([min(yValidate), max(yValidate)], [min(yValidate), max(yValidate)], 'k-', 'LineWidth', 1.5);
xlabel('True y', 'FontWeight', 'bold');
ylabel('Predicted y', 'FontWeight', 'bold');
title('Predictions on Validation Set', 'FontWeight', 'bold');
legend('ML', 'MAP (optimal γ)', 'Perfect', 'Location', 'best');
grid on;
axis equal;

subplot(2, 3, 6);
% Overfitting ratio
overfitting_ratio = mse_val_MAP ./ mse_train_MAP;
semilogx(gamma_values, overfitting_ratio, 'm-', 'LineWidth', 2);
hold on;
semilogx([min(gamma_values), max(gamma_values)], [1, 1], 'k--', 'LineWidth', 1);
semilogx(gamma_optimal, overfitting_ratio(opt_idx), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
xlabel('Prior Variance γ', 'FontWeight', 'bold');
ylabel('Validation/Training MSE', 'FontWeight', 'bold');
title('Overfitting Indicator', 'FontWeight', 'bold');
legend('MAP', 'No overfitting', 'Optimal γ', 'Location', 'best');
grid on;

sgtitle('ML and MAP Estimators Analysis', 'FontWeight', 'bold', 'FontSize', 14);



%% Function definitions (repeated for completeness)
function [xTrain,yTrain,xValidate,yValidate] = hw2q2(Ntrain,Nvalidate)
    data = generateData(Ntrain);
    figure(1), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
    xlabel('x1'),ylabel('x2'), zlabel('y'), title('Training Dataset'),
    xTrain = data(1:2,:); yTrain = data(3,:);

    data = generateData(Nvalidate);
    figure(2), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
    xlabel('x1'),ylabel('x2'), zlabel('y'), title('Validation Dataset'),
    xValidate = data(1:2,:); yValidate = data(3,:);
end

function x = generateData(N)
    gmmParameters.priors = [.3,.4,.3]; % priors should be a row vector
    gmmParameters.meanVectors = [-10 0 10;0 0 0;10 0 -10];
    gmmParameters.covMatrices(:,:,1) = [1 0 -3;0 1 0;-3 0 15];
    gmmParameters.covMatrices(:,:,2) = [8 0 0;0 .5 0;0 0 .5];
    gmmParameters.covMatrices(:,:,3) = [1 0 -3;0 1 0;-3 0 15];
    [x,labels] = generateDataFromGMM(N,gmmParameters);
end

function [x,labels] = generateDataFromGMM(N,gmmParameters)
    priors = gmmParameters.priors;
    meanVectors = gmmParameters.meanVectors;
    covMatrices = gmmParameters.covMatrices;
    n = size(gmmParameters.meanVectors,1);
    C = length(priors);
    x = zeros(n,N); labels = zeros(1,N); 
    u = rand(1,N); thresholds = [cumsum(priors),1];
    for l = 1:C
        indl = find(u <= thresholds(l)); Nl = length(indl);
        labels(1,indl) = l*ones(1,Nl);
        u(1,indl) = 1.1*ones(1,Nl);
        x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
    end
end