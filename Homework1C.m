clear all, close all,
% Generate samples (provided code)
N = 10000; p0 = 0.65; p1 = 0.35;
u = rand(1,N)>=p0; N0 = length(find(u==0)); N1 = length(find(u==1));

% Class 0 samples
mu0 = [-1/2;-1/2;-1/2]; 
Sigma0 = [1,-0.5,0.3;-0.5,1,-0.5;0.3,-0.5,1];
r0 = mvnrnd(mu0, Sigma0, N0);
m0_est = mean(r0)';

c0_est = cov(r0);
fprintf('estimated covariance for class 0:\n');

disp(c0_est);
fprintf('estimated mean for class 0:\n');
disp(m0_est);

% Class 1 samples
mu1 = [1; 1; 1]; 
Sigma1 = [1, 0.3, -0.2; 0.3, 1, 0.3; -0.2, 0.3, 1];
r1 = mvnrnd(mu1, Sigma1, N1);
m1_est = mean(r1)';
c1_est = cov(r1);
fprintf('estimated covariance for class 1:\n');
disp(c1_est);

fprintf('estimated mean for class 1:\n');
disp(m1_est)

Sw = c0_est + c1_est;

fprintf('\nWithin-class scatter matrix Sw:\n');
disp(Sw);

% 2. Compute Between-class Scatter Matrix (Sb)
mean_diff = m0_est - m1_est;
Sb = mean_diff * mean_diff';
fprintf('\nBetween-class scatter matrix Sb:\n');
disp(Sb);

% Using generalized eigendecomposition
[V, D] = eig(Sb, Sw);
[~, idx] = max(real(diag(D)));
w_LDA_eigen = V(:, idx);
% Ensure consistent sign
if w_LDA_eigen' * mean_diff < 0
    w_LDA_eigen = -w_LDA_eigen;
end
w_LDA_eigen = w_LDA_eigen / norm(w_LDA_eigen);

fprintf('\nLDA vector via eigendecomposition:\n');
disp(w_LDA_eigen);
% Project data
y0 = r0 * w_LDA_eigen;
y1 = r1 * w_LDA_eigen;

%% Generate ROC curve by sweeping threshold
tau_array = linspace(min([y0; y1])-1, max([y0; y1])+1, 1000);
TPR = zeros(length(tau_array), 1);
FPR = zeros(length(tau_array), 1);

for i = 1:length(tau_array)
    tau = tau_array(i);
    
    % Decision rule: if w'x >= tau, classify as class 1
    if mean(y1) > mean(y0)
        TPR(i) = sum(y1 >= tau) / N1;  % True Positive Rate
        FPR(i) = sum(y0 >= tau) / N0;  % False Positive Rate
    else
        TPR(i) = sum(y1 <= tau) / N1;
        FPR(i) = sum(y0 <= tau) / N0;
    end
end

%% Find optimal threshold (highest TPR - FPR)
[~, idx_opt] = max(TPR - FPR);
tau_opt = tau_array(idx_opt);
TPR_opt = TPR(idx_opt);
FPR_opt = FPR(idx_opt);

% Calculate probability of error at optimal threshold
if mean(y1) > mean(y0)
    FN_opt = sum(y1 < tau_opt) / N1;  % False Negative Rate
else
    FN_opt = sum(y1 > tau_opt) / N1;
end

P_error_opt = FPR_opt * p0 + FN_opt * p1;

%% Plot ROC curve with optimal operating point
figure('Position', [200, 200, 600, 500]);
plot(FPR, TPR, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);  % Random classifier
plot(FPR_opt, TPR_opt, 'ro', 'MarkerSize', 12, 'LineWidth', 2, 'MarkerFaceColor', 'r');

xlabel('False Positive Rate', 'FontSize', 12);
ylabel('True Positive Rate', 'FontSize', 12);
title('ROC Curve', 'FontSize', 14);
legend('ROC Curve', 'Random Classifier', ...
       sprintf('Optimal Point\n(τ=%.3f, P_{err}=%.3f)', tau_opt, P_error_opt), ...
       'Location', 'southeast', 'FontSize', 11);
grid on;
axis square;
xlim([0 1]); ylim([0 1]);

%% Print results
fprintf('Optimal Operating Point:\n');
fprintf('  Threshold τ = %.4f\n', tau_opt);
fprintf('  TPR = %.4f\n', TPR_opt);
fprintf('  FPR = %.4f\n', FPR_opt);
fprintf('  Probability of Error = %.4f\n', P_error_opt);