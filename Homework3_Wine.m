%% Minimum Probability of Error Classifier - Corrected Variable Names
clear; clc;

%% Load Data with Correct Variable Names
load('wine_data_complete.mat');

% Your file contains: X_all, y_all, X_red, y_red, X_white, y_white, etc.
% Use X_all and y_all for all wines combined
X = X_all;
y = y_all;


fprintf('===== Data Loaded =====\n');
fprintf('Total samples: %d\n', num_samples);
fprintf('Number of features: %d\n', num_features);
fprintf('Feature names: %s\n', strjoin(string(feature_names), ', '));
fprintf('Unique quality scores: %s\n\n', num2str(unique(y)'));

%% Parameters
lambda = 1e-4;  % Regularization parameter

classes = unique(y);
K = length(classes);
D = size(X, 2);
N = size(X, 1);


mu = cell(K, 1);
Sigma = cell(K, 1);
prior = zeros(K, 1);
y_pred = zeros(N, 1);

%% Estimate Parameters for Each Class
fprintf('===== Parameter Estimation =====\n');

for k = 1:K
    idx = (y == classes(k));
    
    % Sample mean
    mu{k} = mean(X(idx, :))';
    
    % Regularized covariance
    Sigma{k} = cov(X(idx, :)) + lambda * eye(D);
    
    % Prior probability
    prior(k) = mean(idx);
    
    fprintf('Class %d (Quality %d): %d samples, prior = %.3f\n', ...
            k, classes(k), sum(idx), prior(k));
end

%% Classification
fprintf('\n===== Classification =====\n');

for i = 1:N
    x = X(i, :)';
    log_post = zeros(K, 1);
    
    for k = 1:K
        diff = x - mu{k};
        log_post(k) = -0.5 * diff' * (Sigma{k} \ diff) ...
                     - 0.5 * log(det(Sigma{k})) ...
                     + log(prior(k));
    end
    
    [~, y_pred(i)] = max(log_post);
end

y_pred = classes(y_pred);

%% Results
accuracy = mean(y_pred == y) * 100;
conf_matrix = confusionmat(y, y_pred);

fprintf('\nOverall Accuracy: %.2f%%\n\n', accuracy);
fprintf('Confusion Matrix:\n');
disp(conf_matrix);

%% Visualize
figure(1);
confusionchart(y, y_pred);
title(sprintf('All Wines - Accuracy: %.2f%%', accuracy));

%% Wine Data Visualization using Feature Subsets

%% Create 3D Scatter Plot Visualization
fprintf('\n===== 3D Visualization - Feature Subsets =====\n');

% Define colors for each class (7 distinct colors)
colors = [
    0.8500 0.3250 0.0980;  % Red-orange
    0.0000 0.4470 0.7410;  % Blue
    0.9290 0.6940 0.1250;  % Yellow
    0.4940 0.1840 0.5560;  % Purple
    0.4660 0.6740 0.1880;  % Green
    0.3010 0.7450 0.9330;  % Cyan
    0.6350 0.0780 0.1840   % Dark red
];

% Create figure
figure('Name', '3D Wine Data Visualization', 'Position', [100 100 1000 800]);
%figure(2)

% Select first 3 features for visualization
% You can change these indices to visualize different feature combinations
feature_idx = [1, 2, 3];  % Change these to explore different features

% Create 3D scatter plot
hold on;



for k = 1:K
    % Get data points for current class
    class_mask = (y == classes(k));
    
    % Plot with different colors and markers
    scatter3(X(class_mask, feature_idx(1)), ...
             X(class_mask, feature_idx(2)), ...
             X(class_mask, feature_idx(3)), ...
             50, colors(k, :), 'filled', 'MarkerEdgeColor', 'k', ...
             'LineWidth', 0.5, 'MarkerFaceAlpha', 0.7);
end
hold off;

% Add labels and formatting
xlabel(sprintf('Feature %d', feature_idx(1)), 'FontSize', 12, 'FontWeight', 'bold');
ylabel(sprintf('Feature %d', feature_idx(2)), 'FontSize', 12, 'FontWeight', 'bold');
zlabel(sprintf('Feature %d', feature_idx(3)), 'FontSize', 12, 'FontWeight', 'bold');
title('3D Scatter Plot of Wine Data (First 3 Features)', 'FontSize', 14, 'FontWeight', 'bold');
view(30,30);

% Add legend
legend_labels = arrayfun(@(x) sprintf('Class %d (Quality %d)', find(classes==x), x), ...
                        classes, 'UniformOutput', false);
legend(legend_labels, 'Location', 'best', 'FontSize', 10);



% Print feature combination info
