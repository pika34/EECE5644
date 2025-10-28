%% Part 2: Logistic Regression (Linear and Quadratic) with Maximum Likelihood Estimation
% Train both logistic-linear and logistic-quadratic classifiers 
% on three different dataset sizes using MLE

%% ============================================================================
% CHECK FOR PART 1 VARIABLES / INITIALIZE IF NEEDED
% ============================================================================


    % Set random seed for reproducibility
    rng(42);
    
    % Define all parameters (SAME as Part 1)
    P_L0 = 0.6;  % P(L=0)
    P_L1 = 0.4;  % P(L=1)
    
    % Gaussian mixture weights
    w01 = 0.5; w02 = 0.5;
    w11 = 0.5; w12 = 0.5;
    
    % Mean vectors (SAME as Part 1)
    m01 = [-0.9; -1.1];   
    m02 = [0.8; 0.75];    
    m11 = [-1.1; 0.9];    
    m12 = [0.9; -0.75];   
    
    % Covariance matrix (SAME as Part 1)
    C = [0.75, 0; 
         0, 1.25];
    
    % Generate datasets using the same function from Part 1

    function [X, y] = generate_data_part2(n_samples, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12)
        n_class0 = binornd(n_samples, P_L0);
        n_class1 = n_samples - n_class0;
        
        X = zeros(n_samples, 2);
        y = zeros(n_samples, 1);
        idx = 1;
        
        for i = 1:n_class0
            if rand() < w01
                X(idx, :) = mvnrnd(m01', C);
            else
                X(idx, :) = mvnrnd(m02', C);
            end
            y(idx) = 0;
            idx = idx + 1;
        end
        
        for i = 1:n_class1
            if rand() < w11
                X(idx, :) = mvnrnd(m11', C);
            else
                X(idx, :) = mvnrnd(m12', C);
            end
            y(idx) = 1;
            idx = idx + 1;
        end
        
        shuffle_idx = randperm(n_samples);
        X = X(shuffle_idx, :);
        y = y(shuffle_idx);
    end
    
    % Generate all datasets with SAME random seeds as Part 1
    rng(42);
    [D50_train, y50_train] = generate_data_part2(50, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);
    rng(43);
    [D500_train, y500_train] = generate_data_part2(500, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);
    rng(44);
    [D5000_train, y5000_train] = generate_data_part2(5000, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);
    rng(45);
    [D10K_validate, y10K_validate] = generate_data_part2(10000, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);
    
    fprintf('Datasets generated successfully!\n\n');

    % Use existing parameters from Part 1
    fprintf('Using datasets and parameters from Part 1\n\n');
    % Parameters should already exist: P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12


%% ============================================================================
% MATHEMATICAL SPECIFICATION
% ============================================================================
fprintf('\n========================================================================\n');
fprintf('LOGISTIC REGRESSION - MAXIMUM LIKELIHOOD ESTIMATION\n');
fprintf('========================================================================\n');
fprintf('Logistic-Linear Model:\n');
fprintf('  h(x,w) = 1/(1 + exp(-w^T z(x)))\n');
fprintf('  z(x) = [1, x₁, x₂]^T  (3 parameters)\n\n');
fprintf('Logistic-Quadratic Model:\n');
fprintf('  h(x,w) = 1/(1 + exp(-w^T z(x)))\n');
fprintf('  z(x) = [1, x₁, x₂, x₁², x₁x₂, x₂²]^T  (6 parameters)\n\n');
fprintf('Objective: Minimize Negative Log-Likelihood\n');
fprintf('  NLL(w) = -Σ[yᵢ log(h(xᵢ,w)) + (1-yᵢ)log(1-h(xᵢ,w))]\n\n');

%% ============================================================================
% FEATURE TRANSFORMATION FUNCTIONS
% ============================================================================

function Z = transform_linear(X)
    % Transform features for logistic-linear model
    % Input: X (N x 2)
    % Output: Z (N x 3) = [1, x₁, x₂]
    N = size(X, 1);
    Z = [ones(N, 1), X];
end

function Z = transform_quadratic(X)
    % Transform features for logistic-quadratic model
    % Input: X (N x 2)
    % Output: Z (N x 6) = [1, x₁, x₂, x₁², x₁x₂, x₂²]
    N = size(X, 1);
    x1 = X(:, 1);
    x2 = X(:, 2);
    Z = [ones(N, 1), x1, x2, x1.^2, x1.*x2, x2.^2];
end

%% ============================================================================
% LOGISTIC REGRESSION CORE FUNCTIONS
% ============================================================================

function p = sigmoid(z)
    % Numerically stable sigmoid function
    p = zeros(size(z));
    pos_mask = z >= 0;
    neg_mask = ~pos_mask;
    
    % For positive values: 1 / (1 + exp(-z))
    p(pos_mask) = 1 ./ (1 + exp(-z(pos_mask)));
    
    % For negative values: exp(z) / (1 + exp(z))
    exp_z = exp(z(neg_mask));
    p(neg_mask) = exp_z ./ (1 + exp_z);
end

function [prob, logits] = logistic_predict_prob(X, w, model_type)
    % Compute probability h(x,w) = P(L=1|X; w)
    % X: N x 2 feature matrix
    % w: parameter vector (3x1 for linear, 6x1 for quadratic)
    % model_type: 'linear' or 'quadratic'
    
    % Transform features
    if strcmp(model_type, 'linear')
        Z = transform_linear(X);
    else  % quadratic
        Z = transform_quadratic(X);
    end
    
    % Compute logits
    logits = Z * w;
    
    % Apply sigmoid
    prob = sigmoid(logits);
end

function nll = negative_log_likelihood(w, X, y, model_type)
    % Compute negative log-likelihood
    % w: parameter vector
    % X: N x 2 feature matrix
    % y: N x 1 binary labels (0 or 1)
    % model_type: 'linear' or 'quadratic'
    
    % Get predicted probabilities
    prob = logistic_predict_prob(X, w, model_type);
    
    % Numerical stability
    epsilon = 1e-10;
    prob = max(min(prob, 1 - epsilon), epsilon);
    
    % Compute NLL
    nll = -sum(y .* log(prob) + (1 - y) .* log(1 - prob));
    
    % Add small L2 regularization
    lambda = 1e-6;
    nll = nll + lambda * sum(w(2:end).^2);  % Don't regularize bias
end

function grad = gradient_nll(w, X, y, model_type)
    % Compute gradient of negative log-likelihood
    
    % Transform features
    if strcmp(model_type, 'linear')
        Z = transform_linear(X);
    else
        Z = transform_quadratic(X);
    end
    
    % Get predicted probabilities
    prob = logistic_predict_prob(X, w, model_type);
    
    % Compute gradient: ∇NLL = -Z^T(y - h)
    grad = -Z' * (y - prob);
    
    % Add L2 regularization gradient
    lambda = 1e-6;
    grad(2:end) = grad(2:end) + 2 * lambda * w(2:end);
end

%% ============================================================================
% TRAINING FUNCTION
% ============================================================================

function [w_opt, nll_final, exitflag, output] = train_logistic_regression(X_train, y_train, model_type, method)
    % Train logistic regression using MLE
    % X_train: N x 2 feature matrix
    % y_train: N x 1 binary labels
    % model_type: 'linear' or 'quadratic'
    % method: 'fminsearch', 'fminunc', or 'gradient_descent'
    
    % Initialize parameters
    if strcmp(model_type, 'linear')
        n_params = 3;
    else
        n_params = 6;
    end
    
    rng(42);  % For reproducibility
    w_init = 0.01 * randn(n_params, 1);
    
    % Define objective function
    obj_fun = @(w) negative_log_likelihood(w, X_train, y_train, model_type);
    
    if strcmp(method, 'fminsearch')
        % Nelder-Mead simplex method (base MATLAB)
        options = optimset('Display', 'final', 'MaxIter', 10000, ...
                          'TolFun', 1e-8, 'TolX', 1e-8);
        [w_opt, nll_final, exitflag, output] = fminsearch(obj_fun, w_init, options);
        
    elseif strcmp(method, 'fminunc')
        % Quasi-Newton method (requires Optimization Toolbox)
        grad_fun = @(w) gradient_nll(w, X_train, y_train, model_type);
        options = optimoptions('fminunc', 'Display', 'final', ...
                              'SpecifyObjectiveGradient', true, ...
                              'MaxIterations', 2000, ...
                              'OptimalityTolerance', 1e-8);
        obj_and_grad = @(w) deal(obj_fun(w), grad_fun(w));
        [w_opt, nll_final, exitflag, output] = fminunc(obj_and_grad, w_init, options);
        
    else  % gradient_descent
        % Custom gradient descent
        [w_opt, nll_final] = gradient_descent_custom(X_train, y_train, w_init, model_type);
        exitflag = 1;
        output.iterations = 0;
    end
end

function [w_opt, nll_final] = gradient_descent_custom(X, y, w_init, model_type)
    % Custom gradient descent with adaptive learning rate
    
    w = w_init;
    learning_rate = 0.1;
    momentum = 0.9;
    max_iter = 10000;
    tol = 1e-8;
    
    velocity = zeros(size(w));
    prev_nll = Inf;
    
    for iter = 1:max_iter
        % Compute gradient
        grad = gradient_nll(w, X, y, model_type);
        
        % Update with momentum
        velocity = momentum * velocity - learning_rate * grad;
        w_new = w + velocity;
        
        % Compute NLL
        nll = negative_log_likelihood(w_new, X, y, model_type);
        
        % Check convergence
        if abs(nll - prev_nll) < tol
            break;
        end
        
        % Adaptive learning rate
        if nll > prev_nll
            learning_rate = learning_rate * 0.5;
            velocity = zeros(size(w));  % Reset momentum
        end
        
        prev_nll = nll;
        w = w_new;
        
        if mod(iter, 1000) == 0
            fprintf('  Iter %d: NLL = %.6f\n', iter, nll);
        end
    end
    
    w_opt = w;
    nll_final = nll;
end

%% ============================================================================
% TRAIN ALL MODELS
% ============================================================================

fprintf('========================================================================\n');
fprintf('TRAINING LOGISTIC REGRESSION MODELS\n');
fprintf('========================================================================\n');

% Choose optimization method
opt_method = 'fminsearch';  % Works in base MATLAB
fprintf('Optimization method: %s\n\n', opt_method);

% Store all models
models = struct();

% Train logistic-linear models
fprintf('TRAINING LOGISTIC-LINEAR MODELS:\n');
fprintf('---------------------------------\n');

fprintf('Training on D50_train (50 samples)...\n');
tic;
[models.linear_50.w, models.linear_50.nll, models.linear_50.exitflag] = ...
    train_logistic_regression(D50_train, y50_train, 'linear', opt_method);
models.linear_50.time = toc;
fprintf('  Time: %.3fs, NLL: %.4f, Parameters: [%.3f, %.3f, %.3f]\n\n', ...
    models.linear_50.time, models.linear_50.nll, models.linear_50.w');

fprintf('Training on D500_train (500 samples)...\n');
tic;
[models.linear_500.w, models.linear_500.nll, models.linear_500.exitflag] = ...
    train_logistic_regression(D500_train, y500_train, 'linear', opt_method);
models.linear_500.time = toc;
fprintf('  Time: %.3fs, NLL: %.4f, Parameters: [%.3f, %.3f, %.3f]\n\n', ...
    models.linear_500.time, models.linear_500.nll, models.linear_500.w');

fprintf('Training on D5000_train (5000 samples)...\n');
tic;
[models.linear_5000.w, models.linear_5000.nll, models.linear_5000.exitflag] = ...
    train_logistic_regression(D5000_train, y5000_train, 'linear', opt_method);
models.linear_5000.time = toc;
fprintf('  Time: %.3fs, NLL: %.4f, Parameters: [%.3f, %.3f, %.3f]\n\n', ...
    models.linear_5000.time, models.linear_5000.nll, models.linear_5000.w');

% Train logistic-quadratic models
fprintf('TRAINING LOGISTIC-QUADRATIC MODELS:\n');
fprintf('------------------------------------\n');

fprintf('Training on D50_train (50 samples)...\n');
tic;
[models.quad_50.w, models.quad_50.nll, models.quad_50.exitflag] = ...
    train_logistic_regression(D50_train, y50_train, 'quadratic', opt_method);
models.quad_50.time = toc;
fprintf('  Time: %.3fs, NLL: %.4f\n', models.quad_50.time, models.quad_50.nll);
fprintf('  Parameters: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n\n', models.quad_50.w');

fprintf('Training on D500_train (500 samples)...\n');
tic;
[models.quad_500.w, models.quad_500.nll, models.quad_500.exitflag] = ...
    train_logistic_regression(D500_train, y500_train, 'quadratic', opt_method);
models.quad_500.time = toc;
fprintf('  Time: %.3fs, NLL: %.4f\n', models.quad_500.time, models.quad_500.nll);
fprintf('  Parameters: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n\n', models.quad_500.w');

fprintf('Training on D5000_train (5000 samples)...\n');
tic;
[models.quad_5000.w, models.quad_5000.nll, models.quad_5000.exitflag] = ...
    train_logistic_regression(D5000_train, y5000_train, 'quadratic', opt_method);
models.quad_5000.time = toc;
fprintf('  Time: %.3fs, NLL: %.4f\n', models.quad_5000.time, models.quad_5000.nll);
fprintf('  Parameters: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n\n', models.quad_5000.w');

%% ============================================================================
% EVALUATION ON VALIDATION SET
% ============================================================================

fprintf('========================================================================\n');
fprintf('EVALUATION ON VALIDATION SET (D10K_validate)\n');
fprintf('========================================================================\n');

function [accuracy, error_rate, conf_matrix, prob_scores, tpr, fpr] = ...
    evaluate_model(X_val, y_val, w, model_type, model_name)
    
    % Get predicted probabilities
    prob_scores = logistic_predict_prob(X_val, w, model_type);
    
    % Make predictions (threshold = 0.5)
    y_pred = double(prob_scores >= 0.5);
    
    % Compute confusion matrix
    conf_matrix = confusionmat(y_val, y_pred);
    if size(conf_matrix, 1) == 2 && size(conf_matrix, 2) == 2
        TN = conf_matrix(1,1);
        FP = conf_matrix(1,2);
        FN = conf_matrix(2,1);
        TP = conf_matrix(2,2);
    else
        % Handle edge cases
        if all(y_pred == 0)
            TN = sum(y_val == 0);
            FP = 0;
            FN = sum(y_val == 1);
            TP = 0;
        else
            TN = 0;
            FP = sum(y_val == 0);
            FN = 0;
            TP = sum(y_val == 1);
        end
    end
    
    % Compute metrics
    accuracy = (TP + TN) / length(y_val);
    error_rate = 1 - accuracy;
    tpr = TP / (TP + FN);  % Sensitivity
    fpr = FP / (FP + TN);  % 1 - Specificity
    
    fprintf('\n%s:\n', model_name);
    fprintf('  Accuracy: %.4f (%.2f%%)\n', accuracy, accuracy*100);
    fprintf('  Error Rate: %.4f (%.2f%%)\n', error_rate, error_rate*100);
    fprintf('  TPR: %.4f, FPR: %.4f\n', tpr, fpr);
end

% Evaluate all models
fprintf('\nLOGISTIC-LINEAR MODELS:\n');
[acc_lin_50, err_lin_50, ~, prob_lin_50] = ...
    evaluate_model(D10K_validate, y10K_validate, models.linear_50.w, 'linear', 'Linear - 50 samples');
[acc_lin_500, err_lin_500, ~, prob_lin_500] = ...
    evaluate_model(D10K_validate, y10K_validate, models.linear_500.w, 'linear', 'Linear - 500 samples');
[acc_lin_5000, err_lin_5000, ~, prob_lin_5000] = ...
    evaluate_model(D10K_validate, y10K_validate, models.linear_5000.w, 'linear', 'Linear - 5000 samples');

fprintf('\nLOGISTIC-QUADRATIC MODELS:\n');
[acc_quad_50, err_quad_50, ~, prob_quad_50] = ...
    evaluate_model(D10K_validate, y10K_validate, models.quad_50.w, 'quadratic', 'Quadratic - 50 samples');
[acc_quad_500, err_quad_500, ~, prob_quad_500] = ...
    evaluate_model(D10K_validate, y10K_validate, models.quad_500.w, 'quadratic', 'Quadratic - 500 samples');
[acc_quad_5000, err_quad_5000, ~, prob_quad_5000] = ...
    evaluate_model(D10K_validate, y10K_validate, models.quad_5000.w, 'quadratic', 'Quadratic - 5000 samples');

%% ============================================================================
% BAYES CLASSIFIER FUNCTIONS (FROM PART 1)
% ============================================================================

function eta = compute_discriminant_score(X, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12)
    % Compute the discriminant score η(x) for Bayes optimal classifier
    % This function is needed for comparison with Bayes optimal
    
    % Compute class-conditional pdfs
    p_x_given_L0 = compute_class_conditional_pdf_bayes(X, 0, m01, m02, m11, m12, C, w01, w02, w11, w12);
    p_x_given_L1 = compute_class_conditional_pdf_bayes(X, 1, m01, m02, m11, m12, C, w01, w02, w11, w12);
    
    % Compute discriminant score with numerical stability
    epsilon = 1e-100;
    eta = (p_x_given_L1 * P_L1) ./ (p_x_given_L0 * P_L0 + epsilon);
end

function pdf_vals = compute_class_conditional_pdf_bayes(X, label, m01, m02, m11, m12, C, w01, w02, w11, w12)
    % Compute p(x|L=label) for Bayes classifier
    
    if label == 0
        pdf1 = w01 * gaussian_pdf_bayes(X, m01, C);
        pdf2 = w02 * gaussian_pdf_bayes(X, m02, C);
    else  % label == 1
        pdf1 = w11 * gaussian_pdf_bayes(X, m11, C);
        pdf2 = w12 * gaussian_pdf_bayes(X, m12, C);
    end
    
    pdf_vals = pdf1 + pdf2;
end

function pdf_vals = gaussian_pdf_bayes(X, mean_vec, cov_mat)
    % Compute multivariate Gaussian PDF for Bayes classifier
    
    [N, d] = size(X);
    mean_vec = mean_vec(:)';  % Ensure row vector
    
    % Compute difference from mean
    diff = X - repmat(mean_vec, N, 1);
    
    % Compute Mahalanobis distance
    cov_inv = inv(cov_mat);
    mahal = sum((diff * cov_inv) .* diff, 2);
    
    % Compute PDF values
    norm_const = 1 / ((2*pi)^(d/2) * sqrt(det(cov_mat)));
    pdf_vals = norm_const * exp(-0.5 * mahal);
end

%% ============================================================================
% ROC CURVES
% ============================================================================

function [fpr, tpr, auc_score] = compute_roc(y_true, scores)
    % Compute ROC curve
    thresholds = unique(sort(scores, 'descend'));
    thresholds = [Inf; thresholds; -Inf];
    
    n_thresh = length(thresholds);
    tpr = zeros(n_thresh, 1);
    fpr = zeros(n_thresh, 1);
    
    for i = 1:n_thresh
        y_pred = double(scores >= thresholds(i));
        cm = confusionmat(y_true, y_pred);
        
        if size(cm, 1) == 2 && size(cm, 2) == 2
            TN = cm(1,1); FP = cm(1,2);
            FN = cm(2,1); TP = cm(2,2);
            tpr(i) = TP / (TP + FN);
            fpr(i) = FP / (FP + TN);
        elseif all(y_pred == 0)
            tpr(i) = 0; fpr(i) = 0;
        else
            tpr(i) = 1; fpr(i) = 1;
        end
    end
    
    % Calculate AUC
    auc_score = abs(trapz(fpr, tpr));
end

% Compute ROC for all models
[fpr_lin_50, tpr_lin_50, auc_lin_50] = compute_roc(y10K_validate, prob_lin_50);
[fpr_lin_500, tpr_lin_500, auc_lin_500] = compute_roc(y10K_validate, prob_lin_500);
[fpr_lin_5000, tpr_lin_5000, auc_lin_5000] = compute_roc(y10K_validate, prob_lin_5000);

[fpr_quad_50, tpr_quad_50, auc_quad_50] = compute_roc(y10K_validate, prob_quad_50);
[fpr_quad_500, tpr_quad_500, auc_quad_500] = compute_roc(y10K_validate, prob_quad_500);
[fpr_quad_5000, tpr_quad_5000, auc_quad_5000] = compute_roc(y10K_validate, prob_quad_5000);

% Bayes optimal for comparison (if eta_scores exists from Part 1)
if exist('eta_scores', 'var') && exist('min_p_error', 'var')
    % Use existing variables from Part 1
    [fpr_bayes, tpr_bayes, auc_bayes] = compute_roc(y10K_validate, eta_scores);
    fprintf('Using Bayes optimal results from Part 1\n');
else
    % Compute eta_scores if not available from Part 1
    fprintf('\nComputing Bayes optimal scores for comparison...\n');
    % Use THE SAME parameters as defined in Part 1
    % These should match exactly with Part 1 definitions
    % P_L0, P_L1, etc. should already be defined from Part 1
    % If not, they are defined at the beginning of this script
    
    eta_scores = compute_discriminant_score(D10K_validate, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);
    [fpr_bayes, tpr_bayes, auc_bayes] = compute_roc(y10K_validate, eta_scores);
    
    % Also compute min_p_error if not available
    if ~exist('min_p_error', 'var')
        y_pred_bayes = double(eta_scores > 1);
        cm_bayes = confusionmat(y10K_validate, y_pred_bayes);
        if size(cm_bayes, 1) == 2 && size(cm_bayes, 2) == 2
            accuracy_bayes = (cm_bayes(1,1) + cm_bayes(2,2)) / length(y10K_validate);
        else
            accuracy_bayes = 0;  % Handle edge case
        end
        min_p_error = 1 - accuracy_bayes;
    end
end

%% ============================================================================
% PLOTTING
% ============================================================================

figure('Position', [100, 100, 1800, 600]);

% -------------------- ROC Curves - Linear Models --------------------
subplot(1, 3, 1);
hold on;
plot(fpr_bayes, tpr_bayes, 'k-', 'LineWidth', 2.5);
plot(fpr_lin_50, tpr_lin_50, 'r--', 'LineWidth', 1.5);
plot(fpr_lin_500, tpr_lin_500, 'g--', 'LineWidth', 1.5);
plot(fpr_lin_5000, tpr_lin_5000, 'b--', 'LineWidth', 1.5);
plot([0, 1], [0, 1], 'k:', 'LineWidth', 1);
hold off;

xlabel('False Positive Rate', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('True Positive Rate', 'FontSize', 11, 'FontWeight', 'bold');
title('ROC Curves - Logistic-Linear Models', 'FontSize', 13, 'FontWeight', 'bold');
legend({sprintf('Bayes (AUC=%.3f)', auc_bayes), ...
        sprintf('Linear 50 (AUC=%.3f)', auc_lin_50), ...
        sprintf('Linear 500 (AUC=%.3f)', auc_lin_500), ...
        sprintf('Linear 5000 (AUC=%.3f)', auc_lin_5000), ...
        'Random'}, 'Location', 'southeast', 'FontSize', 9);
grid on; axis square;
xlim([0, 1]); ylim([0, 1]);

% -------------------- ROC Curves - Quadratic Models --------------------
subplot(1, 3, 2);
hold on;
plot(fpr_bayes, tpr_bayes, 'k-', 'LineWidth', 2.5);
plot(fpr_quad_50, tpr_quad_50, 'r-', 'LineWidth', 1.5);
plot(fpr_quad_500, tpr_quad_500, 'g-', 'LineWidth', 1.5);
plot(fpr_quad_5000, tpr_quad_5000, 'b-', 'LineWidth', 1.5);
plot([0, 1], [0, 1], 'k:', 'LineWidth', 1);
hold off;

xlabel('False Positive Rate', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('True Positive Rate', 'FontSize', 11, 'FontWeight', 'bold');
title('ROC Curves - Logistic-Quadratic Models', 'FontSize', 13, 'FontWeight', 'bold');
legend({sprintf('Bayes (AUC=%.3f)', auc_bayes), ...
        sprintf('Quad 50 (AUC=%.3f)', auc_quad_50), ...
        sprintf('Quad 500 (AUC=%.3f)', auc_quad_500), ...
        sprintf('Quad 5000 (AUC=%.3f)', auc_quad_5000), ...
        'Random'}, 'Location', 'southeast', 'FontSize', 9);
grid on; axis square;
xlim([0, 1]); ylim([0, 1]);

% -------------------- Decision Boundaries --------------------
subplot(1, 3, 3);
hold on;

% Create mesh for visualization
h = 0.05;
x_range = [min(D10K_validate(:,1))-1, max(D10K_validate(:,1))+1];
y_range = [min(D10K_validate(:,2))-1, max(D10K_validate(:,2))+1];
[xx, yy] = meshgrid(x_range(1):h:x_range(2), y_range(1):h:y_range(2));
mesh_points = [xx(:), yy(:)];

% Plot decision boundaries
Z_lin_5000 = logistic_predict_prob(mesh_points, models.linear_5000.w, 'linear');
Z_quad_5000 = logistic_predict_prob(mesh_points, models.quad_5000.w, 'quadratic');
Z_bayes = compute_discriminant_score(mesh_points, P_L0, P_L1, m01, m02, m11, m12, C, w01, w02, w11, w12);

Z_lin_5000 = reshape(Z_lin_5000, size(xx));
Z_quad_5000 = reshape(Z_quad_5000, size(xx));
Z_bayes = reshape(Z_bayes, size(xx));

contour(xx, yy, Z_lin_5000, [0.5, 0.5], 'b--', 'LineWidth', 2);
contour(xx, yy, Z_quad_5000, [0.5, 0.5], 'r-', 'LineWidth', 2);
contour(xx, yy, Z_bayes, [1, 1], 'k-', 'LineWidth', 2.5);

% Plot sample points
n_plot = 500;
idx = randperm(length(y10K_validate), n_plot);
scatter(D10K_validate(idx(y10K_validate(idx)==0), 1), ...
        D10K_validate(idx(y10K_validate(idx)==0), 2), ...
        8, 'b', 'o', 'filled', 'MarkerFaceAlpha', 0.3);
scatter(D10K_validate(idx(y10K_validate(idx)==1), 1), ...
        D10K_validate(idx(y10K_validate(idx)==1), 2), ...
        8, 'r', '^', 'filled', 'MarkerFaceAlpha', 0.3);

xlabel('Feature x₁', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Feature x₂', 'FontSize', 11, 'FontWeight', 'bold');
title('Decision Boundaries (5000 samples)', 'FontSize', 13, 'FontWeight', 'bold');
legend({'Linear', 'Quadratic', 'Bayes'}, 'Location', 'best', 'FontSize', 10);
grid on;
xlim(x_range); ylim(y_range);
axis equal;
hold off;

%% ============================================================================
% ERROR COMPARISON TABLE
% ============================================================================

fprintf('\n========================================================================\n');
fprintf('PERFORMANCE SUMMARY TABLE\n');
fprintf('========================================================================\n');
fprintf('\n                     Error Rates on Validation Set\n');
fprintf('┌─────────────────┬────────────┬────────────┬────────────┐\n');
fprintf('│     Model       │ 50 samples │ 500 samples│5000 samples│\n');
fprintf('├─────────────────┼────────────┼────────────┼────────────┤\n');
fprintf('│ Logistic-Linear │   %.2f%%    │   %.2f%%    │   %.2f%%    │\n', ...
    err_lin_50*100, err_lin_500*100, err_lin_5000*100);
fprintf('│ Logistic-Quad   │   %.2f%%    │   %.2f%%    │   %.2f%%    │\n', ...
    err_quad_50*100, err_quad_500*100, err_quad_5000*100);
fprintf('├─────────────────┼────────────┴────────────┴────────────┤\n');
fprintf('│ Bayes Optimal   │              %.2f%%                    │\n', min_p_error*100);
fprintf('└─────────────────┴───────────────────────────────────┘\n');

fprintf('\n                        AUC Scores\n');
fprintf('┌─────────────────┬────────────┬────────────┬────────────┐\n');
fprintf('│     Model       │ 50 samples │ 500 samples│5000 samples│\n');
fprintf('├─────────────────┼────────────┼────────────┼────────────┤\n');
fprintf('│ Logistic-Linear │   %.4f    │   %.4f    │   %.4f    │\n', ...
    auc_lin_50, auc_lin_500, auc_lin_5000);
fprintf('│ Logistic-Quad   │   %.4f    │   %.4f    │   %.4f    │\n', ...
    auc_quad_50, auc_quad_500, auc_quad_5000);
fprintf('├─────────────────┼────────────┴────────────┴────────────┤\n');
fprintf('│ Bayes Optimal   │              %.4f                   │\n', auc_bayes);
fprintf('└─────────────────┴───────────────────────────────────┘\n');

%% ============================================================================
% FINAL SUMMARY
% ============================================================================

fprintf('\n========================================================================\n');
fprintf('KEY OBSERVATIONS\n');
fprintf('========================================================================\n');
fprintf('1. PERFORMANCE IMPROVEMENT WITH DATA:\n');
fprintf('   - Both models improve with more training data\n');
fprintf('   - Quadratic models generally outperform linear models\n\n');

fprintf('2. MODEL COMPLEXITY:\n');
fprintf('   - Linear: 3 parameters, straight decision boundary\n');
fprintf('   - Quadratic: 6 parameters, curved decision boundary\n');
fprintf('   - Quadratic can better approximate the Bayes boundary\n\n');

fprintf('3. COMPARISON TO BAYES OPTIMAL:\n');
best_linear_err = err_lin_5000;
best_quad_err = err_quad_5000;
fprintf('   - Best Linear Error: %.2f%% (vs Bayes: %.2f%%)\n', ...
    best_linear_err*100, min_p_error*100);
fprintf('   - Best Quadratic Error: %.2f%% (vs Bayes: %.2f%%)\n', ...
    best_quad_err*100, min_p_error*100);
fprintf('   - Gap from Linear to Bayes: %.2f%%\n', ...
    (best_linear_err - min_p_error)*100);
fprintf('   - Gap from Quadratic to Bayes: %.2f%%\n', ...
    (best_quad_err - min_p_error)*100);

fprintf('\n4. LEARNED PARAMETERS (5000 samples):\n');
fprintf('   Linear:    w = [%.3f, %.3f, %.3f]\n', models.linear_5000.w');
fprintf('   Quadratic: w = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n', models.quad_5000.w');


%% ============================================================================
% APPLY THREE APPROXIMATIONS AND ESTIMATE PROBABILITY OF ERROR
% ============================================================================
% This section applies the three trained approximations (50, 500, 5000 samples)
% to classify all samples in D10K_validate and estimates P(error) using counts

fprintf('\n========================================================================\n');
fprintf('APPLYING THREE APPROXIMATIONS TO D10K_validate\n');
fprintf('========================================================================\n');
fprintf('Classification Rule: Choose L=1 if h(x,w) > 0.5, else choose L=0\n');
fprintf('This approximates the minimum P(error) classification rule\n\n');

%% Apply Linear Models (Three Approximations: 50, 500, 5000 samples)
fprintf('LOGISTIC-LINEAR APPROXIMATIONS:\n');
fprintf('----------------------------------------\n');

% === APPROXIMATION 1: Linear model trained on 50 samples ===
prob_lin_50_val = logistic_predict_prob(D10K_validate, models.linear_50.w, 'linear');
y_pred_lin_50 = double(prob_lin_50_val > 0.5);  % Apply classification rule
cm_lin_50 = confusionmat(y10K_validate, y_pred_lin_50);
errors_lin_50 = cm_lin_50(1,2) + cm_lin_50(2,1);  % FP + FN
p_error_lin_50 = errors_lin_50 / length(y10K_validate);  % Estimate P(error)

fprintf('1. Linear Model (50 samples):\n');
fprintf('   Confusion Matrix:\n');
fprintf('                 Predicted\n');
fprintf('                 L=0    L=1\n');
fprintf('   Actual L=0  %5d  %5d\n', cm_lin_50(1,1), cm_lin_50(1,2));
fprintf('          L=1  %5d  %5d\n', cm_lin_50(2,1), cm_lin_50(2,2));
fprintf('   Total Errors: %d out of %d\n', errors_lin_50, length(y10K_validate));
fprintf('   Estimated P(error): %.4f (%.2f%%)\n\n', p_error_lin_50, p_error_lin_50*100);

% === APPROXIMATION 2: Linear model trained on 500 samples ===
prob_lin_500_val = logistic_predict_prob(D10K_validate, models.linear_500.w, 'linear');
y_pred_lin_500 = double(prob_lin_500_val > 0.5);  % Apply classification rule
cm_lin_500 = confusionmat(y10K_validate, y_pred_lin_500);
errors_lin_500 = cm_lin_500(1,2) + cm_lin_500(2,1);  % FP + FN
p_error_lin_500 = errors_lin_500 / length(y10K_validate);  % Estimate P(error)

fprintf('2. Linear Model (500 samples):\n');
fprintf('   Confusion Matrix:\n');
fprintf('                 Predicted\n');
fprintf('                 L=0    L=1\n');
fprintf('   Actual L=0  %5d  %5d\n', cm_lin_500(1,1), cm_lin_500(1,2));
fprintf('          L=1  %5d  %5d\n', cm_lin_500(2,1), cm_lin_500(2,2));
fprintf('   Total Errors: %d out of %d\n', errors_lin_500, length(y10K_validate));
fprintf('   Estimated P(error): %.4f (%.2f%%)\n\n', p_error_lin_500, p_error_lin_500*100);

% === APPROXIMATION 3: Linear model trained on 5000 samples ===
prob_lin_5000_val = logistic_predict_prob(D10K_validate, models.linear_5000.w, 'linear');
y_pred_lin_5000 = double(prob_lin_5000_val > 0.5);  % Apply classification rule
cm_lin_5000 = confusionmat(y10K_validate, y_pred_lin_5000);
errors_lin_5000 = cm_lin_5000(1,2) + cm_lin_5000(2,1);  % FP + FN
p_error_lin_5000 = errors_lin_5000 / length(y10K_validate);  % Estimate P(error)

fprintf('3. Linear Model (5000 samples):\n');
fprintf('   Confusion Matrix:\n');
fprintf('                 Predicted\n');
fprintf('                 L=0    L=1\n');
fprintf('   Actual L=0  %5d  %5d\n', cm_lin_5000(1,1), cm_lin_5000(1,2));
fprintf('          L=1  %5d  %5d\n', cm_lin_5000(2,1), cm_lin_5000(2,2));
fprintf('   Total Errors: %d out of %d\n', errors_lin_5000, length(y10K_validate));
fprintf('   Estimated P(error): %.4f (%.2f%%)\n\n', p_error_lin_5000, p_error_lin_5000*100);

%% Apply Quadratic Models (Three Approximations: 50, 500, 5000 samples)
fprintf('LOGISTIC-QUADRATIC APPROXIMATIONS:\n');
fprintf('----------------------------------------\n');

% === APPROXIMATION 1: Quadratic model trained on 50 samples ===
prob_quad_50_val = logistic_predict_prob(D10K_validate, models.quad_50.w, 'quadratic');
y_pred_quad_50 = double(prob_quad_50_val > 0.5);  % Apply classification rule
cm_quad_50 = confusionmat(y10K_validate, y_pred_quad_50);
errors_quad_50 = cm_quad_50(1,2) + cm_quad_50(2,1);  % FP + FN
p_error_quad_50 = errors_quad_50 / length(y10K_validate);  % Estimate P(error)

fprintf('1. Quadratic Model (50 samples):\n');
fprintf('   Confusion Matrix:\n');
fprintf('                 Predicted\n');
fprintf('                 L=0    L=1\n');
fprintf('   Actual L=0  %5d  %5d\n', cm_quad_50(1,1), cm_quad_50(1,2));
fprintf('          L=1  %5d  %5d\n', cm_quad_50(2,1), cm_quad_50(2,2));
fprintf('   Total Errors: %d out of %d\n', errors_quad_50, length(y10K_validate));
fprintf('   Estimated P(error): %.4f (%.2f%%)\n\n', p_error_quad_50, p_error_quad_50*100);

% === APPROXIMATION 2: Quadratic model trained on 500 samples ===
prob_quad_500_val = logistic_predict_prob(D10K_validate, models.quad_500.w, 'quadratic');
y_pred_quad_500 = double(prob_quad_500_val > 0.5);  % Apply classification rule
cm_quad_500 = confusionmat(y10K_validate, y_pred_quad_500);
errors_quad_500 = cm_quad_500(1,2) + cm_quad_500(2,1);  % FP + FN
p_error_quad_500 = errors_quad_500 / length(y10K_validate);  % Estimate P(error)

fprintf('2. Quadratic Model (500 samples):\n');
fprintf('   Confusion Matrix:\n');
fprintf('                 Predicted\n');
fprintf('                 L=0    L=1\n');
fprintf('   Actual L=0  %5d  %5d\n', cm_quad_500(1,1), cm_quad_500(1,2));
fprintf('          L=1  %5d  %5d\n', cm_quad_500(2,1), cm_quad_500(2,2));
fprintf('   Total Errors: %d out of %d\n', errors_quad_500, length(y10K_validate));
fprintf('   Estimated P(error): %.4f (%.2f%%)\n\n', p_error_quad_500, p_error_quad_500*100);

% === APPROXIMATION 3: Quadratic model trained on 5000 samples ===
prob_quad_5000_val = logistic_predict_prob(D10K_validate, models.quad_5000.w, 'quadratic');
y_pred_quad_5000 = double(prob_quad_5000_val > 0.5);  % Apply classification rule
cm_quad_5000 = confusionmat(y10K_validate, y_pred_quad_5000);
errors_quad_5000 = cm_quad_5000(1,2) + cm_quad_5000(2,1);  % FP + FN
p_error_quad_5000 = errors_quad_5000 / length(y10K_validate);  % Estimate P(error)

fprintf('3. Quadratic Model (5000 samples):\n');
fprintf('   Confusion Matrix:\n');
fprintf('                 Predicted\n');
fprintf('                 L=0    L=1\n');
fprintf('   Actual L=0  %5d  %5d\n', cm_quad_5000(1,1), cm_quad_5000(1,2));
fprintf('          L=1  %5d  %5d\n', cm_quad_5000(2,1), cm_quad_5000(2,2));
fprintf('   Total Errors: %d out of %d\n', errors_quad_5000, length(y10K_validate));
fprintf('   Estimated P(error): %.4f (%.2f%%)\n\n', p_error_quad_5000, p_error_quad_5000*100);

%% Summary of P(error) Estimates from Three Approximations
fprintf('========================================================================\n');
fprintf('SUMMARY: ESTIMATED P(ERROR) FROM THREE APPROXIMATIONS\n');
fprintf('========================================================================\n');
fprintf('Based on counts of decision-truth label pairs on D10K_validate:\n\n');
fprintf('┌─────────────────────┬────────────┬────────────┬────────────┐\n');
fprintf('│ Approximation       │ 50 samples │ 500 samples│5000 samples│\n');
fprintf('├─────────────────────┼────────────┼────────────┼────────────┤\n');
fprintf('│ Logistic-Linear     │   %.2f%%    │   %.2f%%    │   %.2f%%    │\n', ...
    p_error_lin_50*100, p_error_lin_500*100, p_error_lin_5000*100);
fprintf('│ Logistic-Quadratic  │   %.2f%%    │   %.2f%%    │   %.2f%%    │\n', ...
    p_error_quad_50*100, p_error_quad_500*100, p_error_quad_5000*100);
fprintf('└─────────────────────┴────────────┴────────────┴────────────┘\n');

% Compare with Bayes optimal if available
if exist('min_p_error', 'var')
    fprintf('\nFor comparison, Bayes optimal P(error): %.2f%%\n', min_p_error*100);
    fprintf('\nGap from Bayes optimal:\n');
    fprintf('Linear models:    +%.2f%% → +%.2f%% → +%.2f%%\n', ...
        (p_error_lin_50-min_p_error)*100, ...
        (p_error_lin_500-min_p_error)*100, ...
        (p_error_lin_5000-min_p_error)*100);
    fprintf('Quadratic models: +%.2f%% → +%.2f%% → +%.2f%%\n', ...
        (p_error_quad_50-min_p_error)*100, ...
        (p_error_quad_500-min_p_error)*100, ...
        (p_error_quad_5000-min_p_error)*100);
end

fprintf('\nKey Observations:\n');
fprintf('1. P(error) decreases as training set size increases\n');
fprintf('2. Quadratic models achieve lower P(error) than linear models\n');
fprintf('3. All approximations have higher P(error) than Bayes optimal\n');
