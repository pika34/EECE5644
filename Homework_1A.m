clear all, close all,

% Generate samples (provided code)
N = 10000; p0 = 0.65; p1 = 0.35;
u = rand(1,N)>=p0; N0 = length(find(u==0)); N1 = length(find(u==1));

% Class 0 samples
mu0 = [-1/2;-1/2;-1/2]; 
Sigma0 = [1,-0.5,0.3;-0.5,1,-0.5;0.3,-0.5,1];
r0 = mvnrnd(mu0, Sigma0, N0);

% Class 1 samples
mu1 = [1;1;1]; 
Sigma1 = [1,0.3,-0.2;0.3,1,0.3;-0.2,0.3,1];
r1 = mvnrnd(mu1, Sigma1, N1);

% Plot the generated samples
figure(1), 
plot3(r0(:,1),r0(:,2),r0(:,3),'.b'); axis equal, hold on,
plot3(r1(:,1),r1(:,2),r1(:,3),'.r'); 
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
title('Generated Samples: Blue=Class 0, Red=Class 1');
legend('Class 0', 'Class 1');
grid on;

%% IMPLEMENT CLASSIFIER
% Compute determinants and inverses
detSigma0 = det(Sigma0);
detSigma1 = det(Sigma1);
invSigma0 = inv(Sigma0);
invSigma1 = inv(Sigma1);

% Function to compute likelihood ratio for a sample x
computeLR = @(x) exp(-0.5*log(detSigma1/detSigma0) - ...
    0.5*((x-mu1')*invSigma1*(x-mu1')' - (x-mu0')*invSigma0*(x-mu0')'));

% Compute likelihood ratios for all samples
LR_class0 = zeros(N0, 1);
for i = 1:N0
    LR_class0(i) = computeLR(r0(i,:));
end

LR_class1 = zeros(N1, 1);
for i = 1:N1
    LR_class1(i) = computeLR(r1(i,:));
end

%% GENERATE ROC CURVE
% Create array of gamma values from 0 to large value
gamma_values = [0, logspace(-2, 2, 100)]; % 0, then 10^-2 to 10^2

% Initialize arrays for ROC curve
FPR = zeros(length(gamma_values), 1); % P(D=1|L=0)
TPR = zeros(length(gamma_values), 1); % P(D=1|L=1)

% For each gamma threshold
for k = 1:length(gamma_values)
    gamma = gamma_values(k);
    
    % True Positive Rate: P(D=1|L=1)
    % Count how many class 1 samples have LR > gamma
    TPR(k) = sum(LR_class1 > gamma) / N1;
    
    % False Positive Rate: P(D=1|L=0)
    % Count how many class 0 samples have LR > gamma
    FPR(k) = sum(LR_class0 > gamma) / N0;
end

%% PLOT ROC CURVE
figure(2),
plot(FPR, TPR, 'b-', 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1); % Random classifier line
xlabel('False Positive Rate P(D=1|L=0)');
ylabel('True Positive Rate P(D=1|L=1)');
title('ROC Curve for Minimum Expected Risk Classifier');
grid on;
axis([0 1 0 1]);
legend('ROC Curve', 'Random Classifier', 'Location', 'southeast');

% Mark the optimal point (gamma = p0/p1 = 0.65/0.35)
gamma_optimal = p0/p1;
[~, idx_optimal] = min(abs(gamma_values - gamma_optimal)); %stores the index of the gamma_value which is closest to gamma_optimal
gamma_min_error = gamma_values(idx_optimal);
plot(FPR(idx_optimal), TPR(idx_optimal), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(FPR(idx_optimal)+0.05, TPR(idx_optimal), ...
    sprintf('Optimal γ=%.2f', gamma_min_error), 'FontSize', 10);
P_error = FPR(idx_optimal)*p0 + (1-TPR(idx_optimal))*p1; 

%% DISPLAY KEY INFORMATION
fprintf('\n=== ROC CURVE ANALYSIS ===\n');
fprintf('Total samples: %d (Class 0: %d, Class 1: %d)\n', N, N0, N1);
fprintf('Class priors: P(L=0)=%.2f, P(L=1)=%.2f\n', p0, p1);
fprintf('Optimal gamma (min error): %.4f\n', gamma_optimal);
fprintf('\nKey ROC Points:\n');
fprintf('At γ=0: FPR=%.4f, TPR=%.4f (classify everything as class 1)\n', ...
    FPR(1), TPR(1));
fprintf('At γ≈%.2f: FPR=%.4f, TPR=%.4f (optimal)\n', ...
    gamma_min_error, FPR(idx_optimal), TPR(idx_optimal));
fprintf('Minimum P(error) = %.6f at gamma = %.4f\n', P_error, gamma_optimal);
fprintf('At γ=%.2f: FPR=%.4f, TPR=%.4f (high threshold)\n', ...
    gamma_values(end), FPR(end), TPR(end));

%% SAVE DATA FOR NEXT SECTION
% Store gamma values and corresponding probabilities
ROC_data = [gamma_values', FPR, TPR];
save('ROC_results.mat', 'ROC_data', 'gamma_values', 'FPR', 'TPR', ...
    'LR_class0', 'LR_class1', 'r0', 'r1');

fprintf('\nROC data saved to ROC_results.mat for next section\n');
fprintf('Variables saved: gamma_values, FPR, TPR, LR_class0, LR_class1\n');