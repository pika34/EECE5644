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
hold on
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

%Part B



% Generate samples using TRUE distributions
N = 10000; p0 = 0.65; p1 = 0.35;
u = rand(1,N)>=p0; 
N0 = length(find(u==0)); 
N1 = length(find(u==1));

% TRUE parameters (for data generation)
mu0_true = [-1/2;-1/2;-1/2]; 
Sigma0_true = [1,-0.5,0.3;-0.5,1,-0.5;0.3,-0.5,1];
r0 = mvnrnd(mu0_true, Sigma0_true, N0);

mu1_true = [1;1;1]; 
Sigma1_true = [1,0.3,-0.2;0.3,1,0.3;-0.2,0.3,1];
r1 = mvnrnd(mu1_true, Sigma1_true, N1);

% Visualize samples
figure(1), 
plot3(r0(:,1),r0(:,2),r0(:,3),'.b'); axis equal, hold on,
plot3(r1(:,1),r1(:,2),r1(:,3),'.r'); 
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
title('Generated Samples (True Distributions)');
legend('Class 0', 'Class 1');
grid on;

%% NAIVE BAYES CLASSIFIER (INCORRECT MODEL)
% NAIVE ASSUMPTION: Use identity matrices (assume independence)
mu0_naive = mu0_true;  % Means are still correct
mu1_naive = mu1_true;
Sigma0_naive = eye(3);  % WRONG: assume identity
Sigma1_naive = eye(3);  % WRONG: assume identity

% For identity matrices:
detSigma0_naive = 1;
detSigma1_naive = 1;
invSigma0_naive = eye(3);
invSigma1_naive = eye(3);

fprintf('Using NAIVE BAYES (Identity Covariance) assumption...\n\n');

%% COMPUTE LIKELIHOOD RATIOS WITH NAIVE MODEL
% Likelihood ratios for Class 0 samples (using WRONG model)
LR_naive_class0 = zeros(N0, 1);
for i = 1:N0
    x = r0(i,:);
    diff0 = (x - mu0_naive');
    diff1 = (x - mu1_naive');
    
    % With identity matrices, Mahalanobis distance = Euclidean distance squared
    mahal0 = sum(diff0.^2);
    mahal1 = sum(diff1.^2);
    
    logLR = -0.5*log(detSigma1_naive/detSigma0_naive) - 0.5*(mahal1 - mahal0);
    LR_naive_class0(i) = exp(logLR);
end

% Likelihood ratios for Class 1 samples (using WRONG model)
LR_naive_class1 = zeros(N1, 1);
for i = 1:N1
    x = r1(i,:);
    diff0 = (x - mu0_naive');
    diff1 = (x - mu1_naive');
    
    mahal0 = sum(diff0.^2);
    mahal1 = sum(diff1.^2);
    
    logLR = -0.5*log(detSigma1_naive/detSigma0_naive) - 0.5*(mahal1 - mahal0);
    LR_naive_class1(i) = exp(logLR);
end

%% VARY THRESHOLD GAMMA
gamma_values = [0, logspace(-2, 2, 200)];

% Compute TPR and FPR for each gamma
TPR_naive = zeros(length(gamma_values), 1);
FPR_naive = zeros(length(gamma_values), 1);

for k = 1:length(gamma_values)
    gamma = gamma_values(k);
    TPR_naive(k) = sum(LR_naive_class1 > gamma) / N1;
    FPR_naive(k) = sum(LR_naive_class0 > gamma) / N0;
end

%% FIND MINIMUM P(ERROR) EMPIRICALLY (FROM THE DATA/PLOT)
% Calculate P(error) for all gamma values
P_error_naive = p0 * FPR_naive + p1 * (1 - TPR_naive);

% Find gamma that gives MINIMUM P(error) from the ROC data
[P_error_naive_min, idx_naive_min] = min(P_error_naive);
gamma_naive_min = gamma_values(idx_naive_min);
FPR_naive_min = FPR_naive(idx_naive_min);
TPR_naive_min = TPR_naive(idx_naive_min);

%% PLOT NAIVE BAYES ROC CURVE
figure(2),
plot(FPR_naive, TPR_naive, 'b-', 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);

% Mark minimum P(error) point with RED SQUARE
plot(FPR_naive_min, TPR_naive_min, 'rs', 'MarkerSize', 15, ...
    'MarkerFaceColor', 'r', 'LineWidth', 3);

xlabel('P(D=1|L=0; \gamma) - False Positive Rate', 'FontSize', 12);
ylabel('P(D=1|L=1; \gamma) - True Positive Rate', 'FontSize', 12);
title('ROC Curve - Naive Bayes Classifier (Identity Covariances)', 'FontSize', 14);
grid on;
axis([0 1 0 1]);
axis square;
hold off;

% Annotate minimum error point
text(FPR_naive_min + 0.05, TPR_naive_min, ...
    sprintf('Min P(error)\n\\gamma=%.3f\nP(error)=%.4f', ...
    gamma_naive_min, P_error_naive_min), ...
    'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'red');

legend('Naive Bayes ROC', 'Random Classifier', 'Min P(error)', ...
    'Location', 'southeast');

%% DISPLAY RESULTS
fprintf('\n=== NAIVE BAYES CLASSIFIER RESULTS ===\n');
fprintf('Assumption: Covariance matrices are IDENTITY (wrong!)\n');
fprintf('Total samples: %d (Class 0: %d, Class 1: %d)\n', N, N0, N1);
fprintf('Class priors: P(L=0)=%.2f, P(L=1)=%.2f\n\n', p0, p1);

fprintf('EMPIRICAL optimal gamma from ROC data: %.4f\n', gamma_naive_min);
fprintf('  Minimum P(error): %.6f\n', P_error_naive_min);
fprintf('  Operating point: FPR=%.4f, TPR=%.4f\n\n', FPR_naive_min, TPR_naive_min);

fprintf('Key ROC Points:\n');
fprintf('  At γ=0: FPR=%.4f, TPR=%.4f\n', FPR_naive(1), TPR_naive(1));
fprintf('  At γ≈%.2f: FPR=%.4f, TPR=%.4f (empirical optimum)\n', ...
    gamma_naive_min, FPR_naive_min, TPR_naive_min);
fprintf('  At γ=%.1f: FPR=%.4f, TPR=%.4f\n', ...
    gamma_values(end), FPR_naive(end), TPR_naive(end));

%% SAVE NAIVE BAYES RESULTS
save('Naive_Bayes_results.mat', 'gamma_values', 'FPR_naive', 'TPR_naive', ...
    'P_error_naive', 'gamma_naive_min', 'P_error_naive_min', ...
    'LR_naive_class0', 'LR_naive_class1');