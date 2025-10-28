%% Bayes Optimal Classifier Implementation in MATLAB
% Pattern Recognition - Minimum Probability of Error Classifier
% Clear workspace and close all figures
%clear all; close all; clc;

%% ============================================================================
% MATHEMATICAL SPECIFICATION OF CLASSIFIER
% ============================================================================
fprintf('========================================================================\n');
fprintf('BAYES OPTIMAL CLASSIFIER SPECIFICATION\n');
fprintf('========================================================================\n');
fprintf('Decision Rule: L_hat(x) = argmax_L P(L|x) = argmax_L p(x|L)P(L)\n\n');
fprintf('Discriminant Score Function:\n');
fprintf('η(x) = [p(x|L=1)P(L=1)] / [p(x|L=0)P(L=0)]\n');
fprintf('     = [0.4 × (0.5g(x|m11,C) + 0.5g(x|m12,C))] / [0.6 × (0.5g(x|m01,C) + 0.5g(x|m02,C))]\n\n');
fprintf('Optimal Decision Threshold: τ = 1\n');
fprintf('Classification: If η(x) > τ, then L_hat = 1; else L_hat = 0\n\n');

%% ============================================================================
% PARAMETERS
% ============================================================================

% Set random seed for reproducibility
rng(42);

% Class priors
P_L0 = 0.6;  % P(L=0)
P_L1 = 0.4;  % P(L=1)

% Gaussian mixture weights (equal for all components)
w01 = 0.5; w02 = 0.5;
w11 = 0.5; w12 = 0.5;

% Mean vectors for Gaussian components
m01 = [-0.9; -1.1];   % Class 0, component 1
m02 = [0.8; 0.75];    % Class 0, component 2
m11 = [-1.1; 0.9];    % Class 1, component 1
m12 = [0.9; -0.75];   % Class 1, component 2

% Covariance matrix (same for all components)
C = [0.75, 0; 
     0, 1.25];

% Precompute useful quantities
C_inv = inv(C);
C_det = det(C);

fprintf('========================================================================\n');
fprintf('CLASSIFIER PARAMETERS\n');
fprintf('========================================================================\n');
fprintf('Class priors: P(L=0) = %.1f, P(L=1) = %.1f\n', P_L0, P_L1);
fprintf('Covariance matrix determinant: |C| = %.4f\n', C_det);
fprintf('Covariance matrix:\n');
disp(C);
fprintf('Inverse covariance matrix:\n');
disp(C_inv);

%% ============================================================================
% DATA GENERATION FUNCTION
% ============================================================================

function [X, y] = generate_data(n_samples, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12)
    % Generate data from the true distribution
    % Returns X (features) and y (true labels)
    
    % Determine number of samples from each class
    n_class0 = binornd(n_samples, P_L0);
    n_class1 = n_samples - n_class0;
    
    X = zeros(n_samples, 2);
    y = zeros(n_samples, 1);
    
    idx = 1;
    
    % Generate class 0 samples
    for i = 1:n_class0
        % Choose component with probability w01, w02
        if rand() < w01
            X(idx, :) = mvnrnd(m01', C);
        else
            X(idx, :) = mvnrnd(m02', C);
        end
        y(idx) = 0;
        idx = idx + 1;
    end
    
    % Generate class 1 samples
    for i = 1:n_class1
        % Choose component with probability w11, w12
        if rand() < w11
            X(idx, :) = mvnrnd(m11', C);
        else
            X(idx, :) = mvnrnd(m12', C);
        end
        y(idx) = 1;
        idx = idx + 1;
    end
    
    % Shuffle the data
    shuffle_idx = randperm(n_samples);
    X = X(shuffle_idx, :);
    y = y(shuffle_idx);
end

%% ============================================================================
% GENERATE ALL REQUIRED DATASETS
% ============================================================================

fprintf('\n========================================================================\n');
fprintf('GENERATING DATASETS\n');
fprintf('========================================================================\n');

% Generate datasets with different seeds for reproducibility
rng(42);
[D50_train, y50_train] = generate_data(50, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);

rng(43);
[D500_train, y500_train] = generate_data(500, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);

rng(44);
[D5000_train, y5000_train] = generate_data(5000, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);

rng(45);
[D10K_validate, y10K_validate] = generate_data(10000, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);

fprintf('D50_train: %d samples, %d class 0, %d class 1\n', ...
    size(D50_train, 1), sum(y50_train==0), sum(y50_train==1));
fprintf('D500_train: %d samples, %d class 0, %d class 1\n', ...
    size(D500_train, 1), sum(y500_train==0), sum(y500_train==1));
fprintf('D5000_train: %d samples, %d class 0, %d class 1\n', ...
    size(D5000_train, 1), sum(y5000_train==0), sum(y5000_train==1));
fprintf('D10K_validate: %d samples, %d class 0, %d class 1\n', ...
    size(D10K_validate, 1), sum(y10K_validate==0), sum(y10K_validate==1));

%% ============================================================================
% CLASSIFIER IMPLEMENTATION FUNCTIONS
% ============================================================================

function pdf_vals = gaussian_pdf(X, mean_vec, cov_mat)
    % Compute multivariate Gaussian PDF for each sample in X
    % X: N x d matrix of samples
    % mean_vec: d x 1 mean vector
    % cov_mat: d x d covariance matrix
    
    [N, d] = size(X);
    mean_vec = mean_vec(:)';  % Ensure row vector
    
    % Compute difference from mean
    diff = X - repmat(mean_vec, N, 1);
    
    % Compute Mahalanobis distance for all samples
    cov_inv = inv(cov_mat);
    mahal = sum((diff * cov_inv) .* diff, 2);
    
    % Compute PDF values
    norm_const = 1 / ((2*pi)^(d/2) * sqrt(det(cov_mat)));
    pdf_vals = norm_const * exp(-0.5 * mahal);
end

function pdf_vals = compute_class_conditional_pdf(X, label, m01, m02, m11, m12, C, w01, w02, w11, w12)
    % Compute p(x|L=label) for each sample in X
    % This is a mixture of two Gaussians
    
    if label == 0
        % p(x|L=0) = w01 * g(x|m01,C) + w02 * g(x|m02,C)
        pdf1 = w01 * gaussian_pdf(X, m01, C);
        pdf2 = w02 * gaussian_pdf(X, m02, C);
    else  % label == 1
        % p(x|L=1) = w11 * g(x|m11,C) + w12 * g(x|m12,C)
        pdf1 = w11 * gaussian_pdf(X, m11, C);
        pdf2 = w12 * gaussian_pdf(X, m12, C);
    end
    
    pdf_vals = pdf1 + pdf2;
end

function eta = compute_discriminant_score(X, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12)
    % Compute the discriminant score η(x) for Bayes optimal classifier
    % η(x) = [p(x|L=1)P(L=1)] / [p(x|L=0)P(L=0)]
    
    % Compute class-conditional pdfs
    p_x_given_L0 = compute_class_conditional_pdf(X, 0, m01, m02, m11, m12, C, w01, w02, w11, w12);
    p_x_given_L1 = compute_class_conditional_pdf(X, 1, m01, m02, m11, m12, C, w01, w02, w11, w12);
    
    % Compute discriminant score with numerical stability
    epsilon = 1e-100;  % Prevent division by zero
    eta = (p_x_given_L1 * P_L1) ./ (p_x_given_L0 * P_L0 + epsilon);
end

function predictions = bayes_classifier(X, threshold, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12)
    % Bayes optimal classifier
    % X: Input features (n_samples x 2)
    % threshold: Decision threshold (default=1.0 for minimum error)
    % Returns: predictions (array of 0s and 1s)
    
    eta = compute_discriminant_score(X, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);
    predictions = double(eta > threshold);  % Convert logical to double
end

%% ============================================================================
% APPLY CLASSIFIER TO VALIDATION SET
% ============================================================================

fprintf('\n========================================================================\n');
fprintf('APPLYING BAYES CLASSIFIER TO VALIDATION SET\n');
fprintf('========================================================================\n');

% Compute discriminant scores for all validation samples
eta_scores = compute_discriminant_score(D10K_validate, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);

% Make predictions using optimal threshold (τ = 1)
y_pred_optimal = bayes_classifier(D10K_validate, 1.0, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);

% Compute confusion matrix for optimal threshold
conf_matrix = confusionmat(y10K_validate, y_pred_optimal);
TN = conf_matrix(1,1);
FP = conf_matrix(1,2);
FN = conf_matrix(2,1);
TP = conf_matrix(2,2);

% Calculate performance metrics
accuracy = (TP + TN) / length(y10K_validate);
min_p_error = 1 - accuracy;  % Minimum probability of error
sensitivity = TP / (TP + FN);  % True Positive Rate
specificity = TN / (TN + FP);  % True Negative Rate
fpr_optimal = FP / (FP + TN);  % False Positive Rate
tpr_optimal = TP / (TP + FN);  % True Positive Rate

fprintf('\nConfusion Matrix (at optimal threshold τ = 1):\n');
fprintf('              Predicted\n');
fprintf('              L=0    L=1\n');
fprintf('Actual L=0  %5d  %5d\n', TN, FP);
fprintf('       L=1  %5d  %5d\n', FN, TP);
fprintf('\nPerformance Metrics:\n');
fprintf('  Accuracy: %.4f (%.2f%%)\n', accuracy, accuracy*100);
fprintf('  Min-P(error): %.4f (%.2f%%)\n', min_p_error, min_p_error*100);
fprintf('  Sensitivity (TPR): %.4f\n', sensitivity);
fprintf('  Specificity (TNR): %.4f\n', specificity);
fprintf('  False Positive Rate: %.4f\n', fpr_optimal);

%% ============================================================================
% ROC CURVE ANALYSIS
% ============================================================================

fprintf('\n========================================================================\n');
fprintf('ROC CURVE ANALYSIS\n');
fprintf('========================================================================\n');

% Generate ROC curve by varying threshold
thresholds = logspace(-3, 3, 1000);  % Log-spaced thresholds
n_thresh = length(thresholds);
tpr = zeros(n_thresh, 1);
fpr = zeros(n_thresh, 1);

for i = 1:n_thresh
    y_pred = double(eta_scores > thresholds(i));  % Convert logical to double
    cm = confusionmat(y10K_validate, y_pred);
    if size(cm, 1) == 2 && size(cm, 2) == 2
        TN_i = cm(1,1);
        FP_i = cm(1,2);
        FN_i = cm(2,1);
        TP_i = cm(2,2);
        tpr(i) = TP_i / (TP_i + FN_i);
        fpr(i) = FP_i / (FP_i + TN_i);
    elseif all(y_pred == 0)  % All predictions are 0
        tpr(i) = 0;
        fpr(i) = 0;
    else  % All predictions are 1
        tpr(i) = 1;
        fpr(i) = 1;
    end
end

% Add endpoints for complete ROC curve
fpr = [0; fpr; 1];
tpr = [0; tpr; 1];

% Calculate AUC using trapezoidal rule
roc_auc = trapz(fpr, tpr);

fprintf('ROC AUC: %.4f\n', roc_auc);
fprintf('Number of threshold points: %d\n', n_thresh);
fprintf('Optimal operating point (τ=1): TPR=%.4f, FPR=%.4f\n', tpr_optimal, fpr_optimal);

%% ============================================================================
% PLOTTING
% ============================================================================

% Create figure with subplots
figure('Position', [100, 100, 1400, 600]);

% -------------------- ROC Curve --------------------
subplot(1, 2, 1);
plot(fpr, tpr, 'b-', 'LineWidth', 2.5);
hold on;
plot(fpr_optimal, tpr_optimal, 'r*', 'MarkerSize', 20, 'LineWidth', 2);
plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);
fill([fpr; 0], [tpr; 0], 'b', 'FaceAlpha', 0.1);

xlabel('False Positive Rate (1 - Specificity)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('True Positive Rate (Sensitivity)', 'FontSize', 12, 'FontWeight', 'bold');
title('ROC Curve for Bayes Optimal Classifier', 'FontSize', 14, 'FontWeight', 'bold');
legend({sprintf('ROC curve (AUC = %.4f)', roc_auc), ...
        sprintf('Min-P(error) Classifier (τ=1)\nTPR=%.3f, FPR=%.3f\nP(error)=%.3f', ...
                tpr_optimal, fpr_optimal, min_p_error), ...
        'Random Classifier'}, ...
        'Location', 'southeast', 'FontSize', 10);
grid on;
grid minor;
axis square;
xlim([-0.02, 1.02]);
ylim([-0.02, 1.02]);

% Add annotation arrow
annotation('textarrow', [0.25, 0.28], [0.6, 0.55], ...
           'String', 'Optimal Operating Point', ...
           'FontSize', 10, 'Color', 'red', 'FontWeight', 'bold');

% -------------------- Discriminant Score Distribution --------------------
subplot(1, 2, 2);
edges = linspace(0, max(eta_scores), 50);

% Histogram for class 0
histogram(eta_scores(y10K_validate==0), edges, ...
          'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5, ...
          'EdgeColor', 'black', 'LineWidth', 0.5);
hold on;

% Histogram for class 1
histogram(eta_scores(y10K_validate==1), edges, ...
          'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5, ...
          'EdgeColor', 'black', 'LineWidth', 0.5);

% Optimal threshold line
xline(1.0, 'g--', 'LineWidth', 2.5);

xlabel('Discriminant Score η(x)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Probability Density', 'FontSize', 12, 'FontWeight', 'bold');
title('Distribution of Discriminant Scores', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Class 0', 'Class 1', 'Optimal Threshold (τ=1)'}, ...
       'Location', 'northeast', 'FontSize', 10);
grid on;
xlim([0, 5]);

% Add statistics text box
stats_text = sprintf('Statistics:\nMean η(x|L=0): %.3f\nMean η(x|L=1): %.3f\nMedian η(x|L=0): %.3f\nMedian η(x|L=1): %.3f', ...
    mean(eta_scores(y10K_validate==0)), ...
    mean(eta_scores(y10K_validate==1)), ...
    median(eta_scores(y10K_validate==0)), ...
    median(eta_scores(y10K_validate==1)));
text(0.98, 0.98, stats_text, 'Units', 'normalized', ...
     'FontSize', 9, 'VerticalAlignment', 'top', ...
     'HorizontalAlignment', 'right', ...
     'BackgroundColor', 'w', 'EdgeColor', 'k');

%% ============================================================================
% DECISION BOUNDARY VISUALIZATION
% ============================================================================

figure('Position', [100, 100, 800, 700]);

% Create mesh grid for decision boundary
h = 0.02;  % step size in mesh
margin = 1.5;
x_min = min(D10K_validate(:, 1)) - margin;
x_max = max(D10K_validate(:, 1)) + margin;
y_min = min(D10K_validate(:, 2)) - margin;
y_max = max(D10K_validate(:, 2)) + margin;

[xx, yy] = meshgrid(x_min:h:x_max, y_min:h:y_max);

% Compute discriminant scores for mesh
mesh_points = [xx(:), yy(:)];
Z = compute_discriminant_score(mesh_points, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);
Z = reshape(Z, size(xx));

% Plot decision regions
levels = [0, 0.5, 1, 2, 5, 10, 20];
contourf(xx, yy, Z, levels, 'LineColor', 'none');
colormap(flipud(hot));
alpha(0.3);
hold on;

% Plot decision boundary (η = 1)
contour(xx, yy, Z, [1, 1], 'k', 'LineWidth', 3);

% Additional contour lines
contour(xx, yy, Z, [0.5, 2], 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1, 'LineStyle', '--');

% Subsample validation data for plotting
n_plot = min(2000, length(D10K_validate));
plot_idx = randperm(length(D10K_validate), n_plot);
X_plot = D10K_validate(plot_idx, :);
y_plot = y10K_validate(plot_idx);

% Plot data points
scatter(X_plot(y_plot==0, 1), X_plot(y_plot==0, 2), 20, 'b', 'o', ...
        'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'green', 'LineWidth', 0.5);
scatter(X_plot(y_plot==1, 1), X_plot(y_plot==1, 2), 20, 'r', '^', ...
        'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'blue', 'LineWidth', 0.5);

% Plot Gaussian component centers
scatter(m01(1), m01(2), 500, 'b', '*', 'LineWidth', 2);
scatter(m02(1), m02(2), 500, 'b', '*', 'LineWidth', 2);
scatter(m11(1), m11(2), 500, 'r', '*', 'LineWidth', 2);
scatter(m12(1), m12(2), 500, 'r', '*', 'LineWidth', 2);

% Add labels to centers
text(m01(1)-0.2, m01(2)-0.2, 'm_{01}', 'FontSize', 10, 'FontWeight', 'bold');
text(m02(1)+0.1, m02(2)+0.1, 'm_{02}', 'FontSize', 10, 'FontWeight', 'bold');
text(m11(1)-0.2, m11(2)+0.1, 'm_{11}', 'FontSize', 10, 'FontWeight', 'bold');
text(m12(1)+0.1, m12(2)-0.1, 'm_{12}', 'FontSize', 10, 'FontWeight', 'bold');

xlabel('Feature x_1', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Feature x_2', 'FontSize', 12, 'FontWeight', 'bold');
title({'Bayes Optimal Decision Boundary', '(Black curve shows η(x) = 1)'}, ...
      'FontSize', 14, 'FontWeight', 'bold');
legend({'', 'Decision Boundary', '', 'Class 0 (True)', 'Class 1 (True)', ...
        'Class 0 centers', '', 'Class 1 centers', ''}, ...
       'Location', 'best', 'FontSize', 9);
grid on;
axis equal;
xlim([x_min, x_max]);
ylim([y_min, y_max]);

% Add colorbar
c = colorbar;
ylabel(c, 'Discriminant Score η(x)', 'FontSize', 11);

%% ============================================================================
% SUMMARY
% ============================================================================

fprintf('\n========================================================================\n');
fprintf('SUMMARY OF RESULTS\n');
fprintf('========================================================================\n');
fprintf('\nMATHEMATICAL SPECIFICATION:\n');
fprintf('  η(x) = [p(x|L=1)·P(L=1)] / [p(x|L=0)·P(L=0)]\n');
fprintf('  Decision: If η(x) > 1, classify as L=1; else L=0\n');
fprintf('\nVALIDATION RESULTS (D10K_validate):\n');
fprintf('  Total samples: %d\n', length(y10K_validate));
fprintf('  Correct classifications: %d\n', TP + TN);
fprintf('  Misclassifications: %d\n', FP + FN);
fprintf('  Estimated min-P(error): %.4f (%.2f%%)\n', min_p_error, min_p_error*100);
fprintf('  Achieved accuracy: %.4f (%.2f%%)\n', accuracy, accuracy*100);
fprintf('  ROC AUC: %.4f\n', roc_auc);
fprintf('\nThe red star on the ROC curve marks the min-P(error) operating point.\n');

fprintf('This is the theoretically optimal classifier for the given distributions.\n');
