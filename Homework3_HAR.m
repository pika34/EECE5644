%% Human Activity Recognition - Optimized Gaussian Classifier
clear; clc;

%% Load Data
fprintf('Loading HAR data...\n');
tic;
X_train = load('C:\Users\mmali\Downloads\UCI HAR Dataset\UCI HAR Dataset\train\X_train.txt');
y_train = load('C:\Users\mmali\Downloads\UCI HAR Dataset\UCI HAR Dataset\train\y_train.txt');
X_test = load('C:\Users\mmali\Downloads\UCI HAR Dataset\UCI HAR Dataset\test\X_test.txt');
y_test = load('C:\Users\mmali\Downloads\UCI HAR Dataset\UCI HAR Dataset\test\y_test.txt');
fprintf('Data loaded in %.2f seconds\n', toc);

%% Estimate Parameters Using ALL Samples from Each Class
classes = unique(y_train);
K = length(classes);
D = size(X_train, 2);
lambda = 1e-4; % Base regularization

% Preallocate
mu = cell(K, 1);
Sigma = cell(K, 1);
prior = zeros(K, 1);
Sigma_inv = cell(K, 1);
log_det_Sigma = zeros(K, 1);

fprintf('\nEstimating parameters...\n');
tic;

for k = 1:K
    % Get all samples from class k
    idx = (y_train == classes(k));
    X_k = X_train(idx, :);
    
    % 1. Mean vector (sample average)
    mu{k} = mean(X_k)';
    
    % 2. Sample covariance
    C = cov(X_k);
    
    % 3. Check if regularization needed
    min_eig = min(eig(C));
    if min_eig < 1e-6
        fprintf('Class %d: Adding regularization (min_eig = %e)\n', k, min_eig);
        Sigma{k} = C + max(1e-3, abs(min_eig)*10) * eye(D);
    else
        Sigma{k} = C + lambda * eye(D); % Minimal regularization
    end
    
    % 4. PRECOMPUTE inverse and log determinant for faster classification
    Sigma_inv{k} = inv(Sigma{k});
    log_det_Sigma(k) = log(det(Sigma{k}));
    
    % 5. Class prior
    prior(k) = mean(idx);
    fprintf('Class %d: %d samples, prior = %.3f\n', k, sum(idx), prior(k));
end

fprintf('Parameter estimation completed in %.2f seconds\n', toc);

%% FAST Classification Using Precomputed Values
fprintf('\nClassifying %d test samples...\n', length(y_test));
tic;

N_test = length(y_test);
y_pred = zeros(N_test, 1);
log_posteriors = zeros(N_test, K);  % Store all posteriors for analysis

% Progress indicator
fprintf('Progress: ');
for i = 1:N_test
    % Show progress
    if mod(i, 500) == 0
        fprintf('%d/%d...', i, N_test);
    end
    
    x = X_test(i, :)';
    
    for k = 1:K
        d = x - mu{k};
        % Use PRECOMPUTED inverse and determinant (MUCH FASTER!)
        log_posteriors(i, k) = -0.5 * d' * Sigma_inv{k} * d ...
                               - 0.5 * log_det_Sigma(k) ...
                               + log(prior(k));
    end
    
    % Minimum probability of error: choose max posterior
    [~, y_pred(i)] = max(log_posteriors(i, :));
end

y_pred = classes(y_pred);
fprintf('\nClassification completed in %.2f seconds\n', toc);

%% Calculate Errors and Confusion Matrix
fprintf('\n===== CLASSIFICATION RESULTS =====\n');

% Count errors
errors = (y_pred ~= y_test);
num_errors = sum(errors);
num_correct = N_test - num_errors;

% Error rate (minimum probability of error achieved)
error_rate = num_errors / N_test;
accuracy = num_correct / N_test * 100;

fprintf('\nMinimum Probability of Error Results:\n');
fprintf('  Total samples: %d\n', N_test);
fprintf('  Correct classifications: %d\n', num_correct);
fprintf('  Errors: %d\n', num_errors);
fprintf('  Error rate: %.4f (%.2f%%)\n', error_rate, error_rate * 100);
fprintf('  Accuracy: %.2f%%\n', accuracy);

%% Confusion Matrix
conf_matrix = confusionmat(y_test, y_pred);

fprintf('\nConfusion Matrix:\n');
fprintf('True\\Pred\t');
for j = 1:K
    fprintf('%d\t', classes(j));
end
fprintf('\n');

for i = 1:K
    fprintf('%d\t\t', classes(i));
    for j = 1:K
        fprintf('%d\t', conf_matrix(i, j));
    end
    fprintf('\n');
end

%% Per-Class Performance
fprintf('\nPer-Class Performance:\n');
fprintf('Class\tSamples\tCorrect\tErrors\tAccuracy\n');
fprintf('-----\t-------\t-------\t------\t--------\n');

for k = 1:K
    idx = (y_test == classes(k));
    n_class = sum(idx);
    n_correct = sum(y_pred(idx) == classes(k));
    n_errors = n_class - n_correct;
    
    fprintf('%d\t%d\t%d\t%d\t%.2f%%\n', ...
            classes(k), n_class, n_correct, n_errors, ...
            (n_correct/n_class)*100);
end

%% ===== VISUALIZATION: 2D Projections =====
fprintf('\n===== VISUALIZING DATA PROJECTIONS =====\n');

% Combine train and test for visualization
X_all = [X_train; X_test];
y_all = [y_train; y_test];

% Define colors for each class
colors = lines(K);
activity_names = {'Walking', 'Walking Up', 'Walking Down', 'Sitting', 'Standing', 'Laying'};

%% Figure 1: Feature Subsets (2D Projections)
figure('Position', [50, 50, 1400, 800]);
sgtitle('HAR Dataset: 2D Feature Projections', 'FontSize', 14, 'FontWeight', 'bold');

% Select interesting feature pairs (based on feature importance)
feature_pairs = [1, 2;    % tBodyAcc-mean-X vs Y
                 41, 42;   % tGravityAcc-mean-X vs Y
                 81, 82;   % tBodyAccJerk-mean-X vs Y
                 121, 122; % tBodyGyro-mean-X vs Y
                 241, 242; % fBodyAcc-mean-X vs Y
                 503, 504];% fBodyGyroJerkMag-mean vs std

for p = 1:6
    subplot(2, 3, p);
    
    % Plot each class (without MarkerAlpha)
    for k = 1:K
        idx = (y_train == classes(k));
        % Use plot instead of scatter for older MATLAB versions
        plot(X_train(idx, feature_pairs(p,1)), ...
             X_train(idx, feature_pairs(p,2)), ...
             '.', 'Color', colors(k,:), 'MarkerSize', 8);
        hold on;
    end
    
    xlabel(sprintf('Feature %d', feature_pairs(p,1)));
    ylabel(sprintf('Feature %d', feature_pairs(p,2)));
    title(sprintf('Features %d vs %d', feature_pairs(p,1), feature_pairs(p,2)));
    
    if p == 1
        legend(activity_names, 'Location', 'best', 'FontSize', 8);
    end
    grid on;
end

sgtitle('HAR Dataset: 2D Feature Projections');
