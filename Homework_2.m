% 4-Class Gaussian Mixture Model with Equal Priors
% Each class has prior probability = 0.25

clear all; close all;


%Part A

% Parameters
N = 10000;  % Total number of samples
num_classes = 4;

% Class priors (all equal)
priors = [0.25, 0.25, 0.25, 0.25];

% Generate class labels according to prior distribution
% This uses cumulative probabilities to assign classes
u = rand(1, N);
class_labels = zeros(1, N);
class_labels(u < priors(1)) = 1;
class_labels(u >= priors(1) & u < sum(priors(1:2))) = 2;
class_labels(u >= sum(priors(1:2)) & u < sum(priors(1:3))) = 3;
class_labels(u >= sum(priors(1:3))) = 4;

% Count samples in each class
N1 = sum(class_labels == 1);
N2 = sum(class_labels == 2);
N3 = sum(class_labels == 3);
N4 = sum(class_labels == 4);

fprintf('Number of samples per class:\n');
fprintf('Class 1: %d (%.2f%%)\n', N1, 100*N1/N);
fprintf('Class 2: %d (%.2f%%)\n', N2, 100*N2/N);
fprintf('Class 3: %d (%.2f%%)\n', N3, 100*N3/N);
fprintf('Class 4: %d (%.2f%%)\n', N4, 100*N4/N);

% Define parameters for each class
% Class 1: Centered at origin, circular spread
mu1 = [-3; -2];
Sigma1 = [2, 0; 
          0, 1];

% Class 2: Shifted right, horizontally elongated with positive correlation
mu2 = [1; 1];
Sigma2 = [3, 0.5; 
          0.5, 1];


% Class 3: Shifted up, vertically elongated with negative correlation
mu3 = [5; 0];
Sigma3 = [4, 2; 
          2, 2];

% Class 4: Upper right corner, small variance, slight positive correlation
mu4 = [0; 5];
Sigma4 = [0.5, 0.1; 
          0.1, 0.5];

% Generate samples for each class
r1 = mvnrnd(mu1', Sigma1, N1);
r2 = mvnrnd(mu2', Sigma2, N2);
r3 = mvnrnd(mu3', Sigma3, N3);
r4 = mvnrnd(mu4', Sigma4, N4);

% Combine all samples (optional - for certain analyses)
all_samples = [r1; r2; r3; r4];
all_labels = [ones(N1,1); 2*ones(N2,1); 3*ones(N3,1); 4*ones(N4,1)];

% Plotting
figure(1);
subplot(1,2,1);

% Plot samples from each class
plot(r1(:,1), r1(:,2), '.b', 'MarkerSize', 4); hold on;
plot(r2(:,1), r2(:,2), '.r', 'MarkerSize', 4);
plot(r3(:,1), r3(:,2), '.g', 'MarkerSize', 4);
plot(r4(:,1), r4(:,2), '.m', 'MarkerSize', 4);
title('True Labels');
xlabel('x_1'); ylabel('x_2');


%% Classification using MAP Decision Rule
fprintf('\n===== CLASSIFICATION USING MAP DECISION RULE =====\n');

% Store mean and covariance for each class
mu = {mu1, mu2, mu3, mu4};
Sigma = {Sigma1, Sigma2, Sigma3, Sigma4};

% Initialize predicted labels array
predicted_labels = zeros(N, 1);

% Classify each sample using MAP rule
fprintf('Classifying %d samples...\n', N);

% We need to reorganize the samples to match the original class_labels order
% Currently all_samples has all class 1, then all class 2, etc.
% We need to match the original random order in class_labels

% Create properly ordered samples array
ordered_samples = zeros(N, 2);
idx1 = 1; idx2 = 1; idx3 = 1; idx4 = 1;

for i = 1:N
    if class_labels(i) == 1
        ordered_samples(i, :) = r1(idx1, :);
        idx1 = idx1 + 1;
    elseif class_labels(i) == 2
        ordered_samples(i, :) = r2(idx2, :);
        idx2 = idx2 + 1;
    elseif class_labels(i) == 3
        ordered_samples(i, :) = r3(idx3, :);
        idx3 = idx3 + 1;
    else
        ordered_samples(i, :) = r4(idx4, :);
        idx4 = idx4 + 1;
    end
end

% Now classify each sample
for i = 1:N
    x = ordered_samples(i, :)';  % Current sample as column vector
    
    % Calculate log-likelihood for each class (more numerically stable)
    log_likelihoods = zeros(4, 1);
    
    for j = 1:4
        % Calculate log p(x|L=j)
        diff = x - mu{j};
        log_likelihoods(j) = -0.5 * (diff' * inv(Sigma{j}) * diff) ...
                             -0.5 * log(det(Sigma{j})) ...
                             -log(2*pi);
    end
    
    % Since priors are equal, MAP decision is based on maximum likelihood
    [~, predicted_labels(i)] = max(log_likelihoods);
end

fprintf('Classification complete.\n');

%% Compute Confusion Matrix Using Counters

% Initialize count matrix
confusion_count = zeros(4, 4);

% Count each (predicted, true) pair
for i = 1:N
    true_class = class_labels(i);
    predicted_class = predicted_labels(i);
    confusion_count(predicted_class, true_class) = ...
        confusion_count(predicted_class, true_class) + 1;
end

% Display count matrix
fprintf('\n===== CONFUSION MATRIX =====\n');
fprintf('Rows = Predicted, Columns = True\n\n');
fprintf('          L=1      L=2      L=3      L=4\n');
for i = 1:4
    fprintf('D=%d   ', i);
    for j = 1:4
        fprintf('%6d   ', confusion_count(i, j));
    end
    fprintf('\n');
end

% Convert counts to probabilities: P(D=i|L=j)
confusion_matrix = zeros(4, 4);
for j = 1:4
    class_j_count = sum(class_labels == j);
    if class_j_count > 0
        confusion_matrix(:, j) = confusion_count(:, j) / class_j_count;
    end
end

% Display confusion matrix
fprintf('\n===== Normalized CONFUSION MATRIX P(D=i|L=j) =====\n');
fprintf('Rows = Predicted (D), Columns = True (L)\n\n');
fprintf('          L=1      L=2      L=3      L=4\n');
for i = 1:4
    fprintf('D=%d   ', i);
    for j = 1:4
        fprintf('%7.4f  ', confusion_matrix(i, j));
    end
    fprintf('\n');
end

% Verify each column sums to 1
fprintf('\nColumn sums (should all be 1.0):\n');
for j = 1:4
    col_sum = sum(confusion_matrix(:, j));
    fprintf('Column %d: %.6f\n', j, col_sum);
end

%% Calculate Performance Metrics

fprintf('\n===== PERFORMANCE METRICS =====\n');

% Overall accuracy (using counts)
correct_predictions = sum(diag(confusion_count));
overall_accuracy = correct_predictions / N;
fprintf('Overall Accuracy: %.2f%% (%d/%d correct)\n', ...
    overall_accuracy * 100, correct_predictions, N);

% Per-class metrics
fprintf('\nPer-Class Performance:\n');
fprintf('Class   TPR    Correct/Total\n');
for j = 1:4
    tpr = confusion_matrix(j, j);  % True Positive Rate
    class_count = sum(class_labels == j);
    correct_class = confusion_count(j, j);
    fprintf('  %d    %.4f   %d/%d\n', j, tpr, correct_class, class_count);
end

% Empirical error rate
empirical_error = 1 - overall_accuracy;
fprintf('\nEmpirical Error Rate: %.4f\n', empirical_error);

% Display misclassification details
fprintf('\n===== MISCLASSIFICATION ANALYSIS =====\n');
for i = 1:4
    for j = 1:4
        if i ~= j && confusion_count(i, j) > 0
            fprintf('Class %d misclassified as Class %d: %d times (%.2f%%)\n', ...
                j, i, confusion_count(i, j), confusion_matrix(i, j) * 100);
        end
    end
end

%%  Part 2Visualization of Classification Results

% Add to existing figure
subplot(1,2,2);
% Plot with predicted labels
gscatter(ordered_samples(:,1), ordered_samples(:,2), predicted_labels, 'brgm', '....');
title('Predicted Labels (MAP Classifier)');
xlabel('x_1'); ylabel('x_2');
axis equal; grid on;
legend('Pred Class 1', 'Pred Class 2', 'Pred Class 3', 'Pred Class 4', 'Location', 'best');

% Create new figure for confusion matrix visualization and misclassifications
figure(2);

% Subplot 1: Confusion matrix heatmap
subplot(1,2,1);
imagesc(confusion_matrix);
colorbar;
colormap(flipud(gray));
title('Normalized Confusion Matrix P(D=i|L=j)');
xlabel('True Label (L)');
ylabel('Predicted Label (D)');
set(gca, 'XTick', 1:4, 'YTick', 1:4);
% Add text values
for i = 1:4
    for j = 1:4
        if confusion_matrix(i, j) > 0.5
            text_color = 'white';
        else
            text_color = 'black';
        end
        text(j, i, sprintf('%.3f', confusion_matrix(i,j)), ...
            'HorizontalAlignment', 'center', 'Color', text_color, 'FontWeight', 'bold');
    end
end

% Subplot 2: Show misclassified samples
subplot(1,2,2);
misclassified_idx = find(predicted_labels ~= class_labels');
correctly_classified_idx = find(predicted_labels == class_labels');

% Plot correctly classified in light colors
gscatter(ordered_samples(correctly_classified_idx,1), ...
         ordered_samples(correctly_classified_idx,2), ...
         class_labels(correctly_classified_idx), [0.7 0.7 1; 1 0.7 0.7; 0.7 1 0.7; 1 0.7 1], '.');
hold on;

% Highlight misclassified samples
if ~isempty(misclassified_idx)
    plot(ordered_samples(misclassified_idx,1), ...
         ordered_samples(misclassified_idx,2), ...
         'ko', 'MarkerSize', 4, 'LineWidth', 1);
end

% Plot means
plot(mu1(1), mu1(2), 'b*', 'MarkerSize', 12, 'LineWidth', 2);
plot(mu2(1), mu2(2), 'r*', 'MarkerSize', 12, 'LineWidth', 2);
plot(mu3(1), mu3(2), 'g*', 'MarkerSize', 12, 'LineWidth', 2);
plot(mu4(1), mu4(2), 'm*', 'MarkerSize', 12, 'LineWidth', 2);

title(sprintf('Misclassified Samples (Black circles): %d/%d', length(misclassified_idx), N));
xlabel('x_1'); ylabel('x_2');
axis equal; grid on;
legend('True Class 1', 'True Class 2', 'True Class 3', 'True Class 4', ...
       'Misclassified', 'μ_1', 'μ_2', 'μ_3', 'μ_4', 'Location', 'best');




%% Part 3 Create Visualization with Marker Shapes and Colors
% Marker shapes for true labels, colors for correct/incorrect

figure(3);

% Define marker shapes for each true class
marker_shapes = {'o', 's', '^', 'd'};  % circle, square, triangle, diamond
class_names = {'Class 1 (circle)', 'Class 2 (square)', 'Class 3 (triangle)', 'Class 4 (diamond)'};

% Determine which samples are correctly/incorrectly classified
is_correct = (predicted_labels == class_labels');

% Plot each class with appropriate markers and colors
hold on;
for true_class = 1:4
    % Find indices for this true class
    class_indices = find(class_labels == true_class);
    
    % Separate correct and incorrect classifications
    correct_idx = class_indices(predicted_labels(class_indices) == true_class);
    incorrect_idx = class_indices(predicted_labels(class_indices) ~= true_class);
    
    % Plot correctly classified (green)
    if ~isempty(correct_idx)
        plot(ordered_samples(correct_idx, 1), ordered_samples(correct_idx, 2), ...
             marker_shapes{true_class}, 'MarkerEdgeColor', 'g', ...
             'MarkerFaceColor', 'g', 'MarkerSize', 4, ...
             'DisplayName', [class_names{true_class} ' - Correct']);
    end
    
    % Plot incorrectly classified (red)
    if ~isempty(incorrect_idx)
        plot(ordered_samples(incorrect_idx, 1), ordered_samples(incorrect_idx, 2), ...
             marker_shapes{true_class}, 'MarkerEdgeColor', 'r', ...
             'MarkerFaceColor', 'r', 'MarkerSize', 4, ...
             'DisplayName', [class_names{true_class} ' - Incorrect']);
    end
end

% Plot class means with black stars
plot(mu1(1), mu1(2), 'k*', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', 'Class Means');
plot(mu2(1), mu2(2), 'k*', 'MarkerSize', 15, 'LineWidth', 2, 'HandleVisibility', 'off');
plot(mu3(1), mu3(2), 'k*', 'MarkerSize', 15, 'LineWidth', 2, 'HandleVisibility', 'off');
plot(mu4(1), mu4(2), 'k*', 'MarkerSize', 15, 'LineWidth', 2, 'HandleVisibility', 'off');

% Add labels for means
text(mu1(1)+0.3, mu1(2), 'μ₁', 'FontSize', 10, 'FontWeight', 'bold');
text(mu2(1)+0.3, mu2(2), 'μ₂', 'FontSize', 10, 'FontWeight', 'bold');
text(mu3(1)+0.3, mu3(2), 'μ₃', 'FontSize', 10, 'FontWeight', 'bold');
text(mu4(1)+0.3, mu4(2), 'μ₄', 'FontSize', 10, 'FontWeight', 'bold');

grid on;
axis equal;
xlabel('x₁');
ylabel('x₂');
title(sprintf('Classification Results: Green=Correct, Red=Incorrect\n(Accuracy: %.1f%%)', overall_accuracy*100));

% Create custom legend
legend('Location', 'bestoutside', 'FontSize', 8);

% Add summary text
num_correct = sum(is_correct);
num_incorrect = N - num_correct;
text_str = sprintf('Correct: %d\nIncorrect: %d', num_correct, num_incorrect);
xlims = xlim; ylims = ylim;
text(xlims(1) + 0.02*(xlims(2)-xlims(1)), ...
     ylims(2) - 0.05*(ylims(2)-ylims(1)), ...
     text_str, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
     'FontWeight', 'bold');

hold off;



%Part B

   




%% ERM Classification with Asymmetric Loss Matrix

fprintf('\n===== ERM CLASSIFICATION WITH ASYMMETRIC LOSS =====\n');

% Define the loss matrix
Lambda = [0   10  10  100;
          1   0   10  100;
          1   1   0   100;
          1   1   1   0];

fprintf('Loss Matrix:\n');
disp(Lambda);

%  mean and covariance for each class
mu = {mu1, mu2, mu3, mu4};
Sigma = {Sigma1, Sigma2, Sigma3, Sigma4};

% Initialize predicted labels array for ERM
predicted_labels_ERM = zeros(N, 1);

% Classify each sample using ERM decision rule
fprintf('Classifying %d samples using ERM...\n', N);

for i = 1:N
    x = ordered_samples(i, :)';  % Current sample as column vector
    
    % Calculate likelihood for each class
    likelihoods = zeros(4, 1);
    for j = 1:4
        likelihoods(j) = mvnpdf(x', mu{j}', Sigma{j});
    end
    
    % Calculate expected loss for each possible decision
    expected_loss = zeros(4, 1);
    for decision = 1:4
        % Expected loss = sum over true classes of Lambda(decision,j) * P(L=j|x)
        % With equal priors P(L=j) = 0.25
        for true_class = 1:4
            % P(L=j|x) proportional to p(x|L=j) * P(L=j)
            posterior_prop = likelihoods(true_class) * priors(true_class);
            expected_loss(decision) = expected_loss(decision) + ...
                Lambda(decision, true_class) * posterior_prop;
        end
    end
    
    % Make decision: choose class with minimum expected loss
    [~, predicted_labels_ERM(i)] = min(expected_loss);
end

fprintf('ERM Classification complete.\n');

%%  Calculate Minimum Expected Risk

fprintf('\n===== CALCULATING MINIMUM EXPECTED RISK =====\n');

% Initialize risk calculation variables
total_risk = 0;
loss_details = []; % Store individual losses for analysis

% Process each sample
for i = 1:N
    % Get true and predicted labels
    true_label = class_labels(i);
    predicted_label = predicted_labels_ERM(i);
    
    % Look up the loss for this prediction
    sample_loss = Lambda(predicted_label, true_label);
    
    % Add to total risk
    total_risk = total_risk + sample_loss;
    
    % Store details for analysis
    loss_details(i) = sample_loss;
end

% Calculate average risk (this is the minimum expected risk estimate)
minimum_expected_risk = total_risk / N;

fprintf('Total loss over %d samples: %.0f\n', N, total_risk);
fprintf('MINIMUM EXPECTED RISK (Sample Average): %.4f\n\n', minimum_expected_risk);



