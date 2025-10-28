clear; close all; clc;

%% Configuration Parameters
% True vehicle position (must be inside unit circle)
x_true = 0.3;  % True x position
y_true = 0.4;  % True y position

% Verify vehicle is inside unit circle
if sqrt(x_true^2 + y_true^2) > 1
    error('True vehicle position must be inside unit circle');
end

% Measurement noise standard deviation
sigma_measurement = 0.3;

% Prior standard deviations
sigma_x = 0.25;
sigma_y = 0.25;

% Grid parameters for contour plots
grid_range = linspace(-2, 2, 200);
[X_grid, Y_grid] = meshgrid(grid_range, grid_range);

% Set random seed for reproducibility
rng(42);

%% Main loop for different K values
figure('Position', [100, 100, 1200, 900]);
K_values = [1, 2, 3, 4];

% Store all objective values to find common contour levels
all_objectives = cell(4, 1);

for idx = 1:length(K_values)
    K = K_values(idx);
    
    % Generate landmarks evenly spaced on unit circle
    angles = linspace(0, 2*pi, K+1);
    angles = angles(1:end-1);  % Remove last point (same as first)
    landmarks_x = cos(angles);
    landmarks_y = sin(angles);
    
    % Generate range measurements with noise
    measurements = generateRangeMeasurements(x_true, y_true, ...
        landmarks_x, landmarks_y, sigma_measurement);
    
    % Compute MAP objective function over grid
    objective = zeros(size(X_grid));
    for i = 1:size(X_grid, 1)
        for j = 1:size(X_grid, 2)
            objective(i, j) = computeMAPObjective(X_grid(i, j), Y_grid(i, j), ...
                landmarks_x, landmarks_y, measurements, ...
                sigma_measurement, sigma_x, sigma_y);
        end
    end
    
    % Store objectives for finding common levels
    all_objectives{idx} = objective;
    
    % Create subplot
    subplot(2, 2, idx);
    hold on;
    
    % We'll plot contours after computing common levels
    % For now, store the data
end

%% Find common contour levels across all K values
% Combine all objective values and find appropriate levels
all_obj_values = [];
for idx = 1:4
    all_obj_values = [all_obj_values; all_objectives{idx}(:)];
end

% Remove infinite values and compute percentiles for contour levels
all_obj_values(isinf(all_obj_values)) = [];
min_val = min(all_obj_values);
max_val = quantile(all_obj_values, 0.95); % Use 95th percentile as max to avoid outliers

% Create 20 contour levels
num_contours = 20;
contour_levels = linspace(min_val, max_val, num_contours);

%% Plot contours with common levels
for idx = 1:length(K_values)
    K = K_values(idx);
    
    % Generate landmarks again
    angles = linspace(0, 2*pi, K+1);
    angles = angles(1:end-1);
    landmarks_x = cos(angles);
    landmarks_y = sin(angles);
    
    % Select subplot
    subplot(2, 2, idx);
    hold on;
    
    % Plot contours with common levels
    contour(X_grid, Y_grid, all_objectives{idx}, contour_levels, 'LineWidth', 1);
    colormap(jet);
    
    % Plot unit circle (boundary)
    theta_circle = linspace(0, 2*pi, 100);
    plot(cos(theta_circle), sin(theta_circle), 'k--', 'LineWidth', 1.5);
    
    % Plot landmarks
    plot(landmarks_x, landmarks_y, 'ko', 'MarkerSize', 10, ...
        'MarkerFaceColor', 'yellow', 'LineWidth', 2);
    
    % Plot true vehicle position
    plot(x_true, y_true, 'r+', 'MarkerSize', 15, 'LineWidth', 3);
    
    % Find and plot MAP estimate (minimum of objective)
    [min_obj, min_idx] = min(all_objectives{idx}(:));
    [min_i, min_j] = ind2sub(size(all_objectives{idx}), min_idx);
    x_map = X_grid(min_i, min_j);
    y_map = Y_grid(min_i, min_j);
    plot(x_map, y_map, 'g*', 'MarkerSize', 12, 'LineWidth', 2);
    
    % Labels and formatting
    xlabel('X');
    ylabel('Y');
    title(sprintf('K = %d Landmarks', K));
    axis equal;
    axis([-2 2 -2 2]);
    grid on;
    
    % Add legend
    if idx == 1
        legend('Objective Contours', 'Unit Circle', 'Landmarks', ...
            'True Position', 'MAP Estimate', 'Location', 'best');
    end
    
    hold off;
end

% Add overall title
sgtitle(sprintf('MAP Objective Contours (σ_{meas} = %.2f, σ_x = %.2f, σ_y = %.2f)', ...
    sigma_measurement, sigma_x, sigma_y), 'FontSize', 14, 'FontWeight', 'bold');

% Add colorbar to the last subplot
subplot(2, 2, 4);
h = colorbar;
ylabel(h, 'Objective Function Value');

%% Display statistics
fprintf('\n=== MAP Estimation Results ===\n');
fprintf('True Vehicle Position: (%.3f, %.3f)\n', x_true, y_true);
fprintf('Prior Parameters: σ_x = %.3f, σ_y = %.3f\n', sigma_x, sigma_y);
fprintf('Measurement Noise: σ = %.3f\n\n', sigma_measurement);

for idx = 1:length(K_values)
    K = K_values(idx);
    [min_obj, min_idx] = min(all_objectives{idx}(:));
    [min_i, min_j] = ind2sub(size(all_objectives{idx}), min_idx);
    x_map = X_grid(min_i, min_j);
    y_map = Y_grid(min_i, min_j);
    error_dist = sqrt((x_map - x_true)^2 + (y_map - y_true)^2);
    
    fprintf('K = %d: MAP Estimate = (%.3f, %.3f), Error = %.3f\n', ...
        K, x_map, y_map, error_dist);
end

%% Function Definitions

function measurements = generateRangeMeasurements(x_true, y_true, ...
    landmarks_x, landmarks_y, sigma)
    % Generate noisy range measurements
    K = length(landmarks_x);
    measurements = zeros(K, 1);
    
    for i = 1:K
        % Compute true distance
        true_dist = sqrt((x_true - landmarks_x(i))^2 + ...
                        (y_true - landmarks_y(i))^2);
        
        % Add Gaussian noise, ensure non-negative
        noisy_measurement = -1;
        while noisy_measurement < 0
            noise = sigma * randn();
            noisy_measurement = true_dist + noise;
        end
        
        measurements(i) = noisy_measurement;
    end
end

function objective = computeMAPObjective(x, y, landmarks_x, landmarks_y, ...
    measurements, sigma_meas, sigma_x, sigma_y)
    % Compute MAP objective function value at position (x, y)
    
    K = length(landmarks_x);
    objective = 0;
    
    % Measurement term (likelihood)
    for i = 1:K
        dist = sqrt((x - landmarks_x(i))^2 + (y - landmarks_y(i))^2);
        residual = measurements(i) - dist;
        objective = objective + (residual / sigma_meas)^2;
    end
    
    % Prior term
    objective = objective + (x / sigma_x)^2 + (y / sigma_y)^2;
end

%% Additional Analysis: Effect of Number of Landmarks
figure('Position', [100, 100, 800, 600]);

% Run multiple trials to see how K affects estimation accuracy
num_trials = 100;
errors = zeros(length(K_values), num_trials);

for trial = 1:num_trials
    % Generate new random true position inside unit circle
    angle_rand = 2*pi*rand();
    radius_rand = sqrt(rand());  % sqrt for uniform distribution in circle
    x_true_trial = radius_rand * cos(angle_rand);
    y_true_trial = radius_rand * sin(angle_rand);
    
    for idx = 1:length(K_values)
        K = K_values(idx);
        
        % Generate landmarks
        angles = linspace(0, 2*pi, K+1);
        angles = angles(1:end-1);
        landmarks_x = cos(angles);
        landmarks_y = sin(angles);
        
        % Generate measurements
        measurements = generateRangeMeasurements(x_true_trial, y_true_trial, ...
            landmarks_x, landmarks_y, sigma_measurement);
        
        % Find MAP estimate (simplified grid search)
        grid_fine = linspace(-2, 2, 100);
        [X_fine, Y_fine] = meshgrid(grid_fine, grid_fine);
        objective_fine = zeros(size(X_fine));
        
        for i = 1:size(X_fine, 1)
            for j = 1:size(X_fine, 2)
                objective_fine(i, j) = computeMAPObjective(X_fine(i, j), Y_fine(i, j), ...
                    landmarks_x, landmarks_y, measurements, ...
                    sigma_measurement, sigma_x, sigma_y);
            end
        end
        
        % Find minimum
        [~, min_idx] = min(objective_fine(:));
        [min_i, min_j] = ind2sub(size(objective_fine), min_idx);
        x_map = X_fine(min_i, min_j);
        y_map = Y_fine(min_i, min_j);
        
        % Compute error
        errors(idx, trial) = sqrt((x_map - x_true_trial)^2 + ...
                                 (y_map - y_true_trial)^2);
    end
end

% Plot error statistics
subplot(2, 1, 1);
boxplot(errors', K_values);
xlabel('Number of Landmarks (K)');
ylabel('Localization Error');
title('Localization Error vs Number of Landmarks');
grid on;

subplot(2, 1, 2);
mean_errors = mean(errors, 2);
std_errors = std(errors, 0, 2);
errorbar(K_values, mean_errors, std_errors, 'b-o', 'LineWidth', 2);
xlabel('Number of Landmarks (K)');
ylabel('Mean Localization Error');
title(sprintf('Mean Error with Standard Deviation (%d trials)', num_trials));
grid on;

fprintf('\n=== Statistical Analysis ===\n');
for idx = 1:length(K_values)
    fprintf('K = %d: Mean Error = %.4f ± %.4f\n', ...
        K_values(idx), mean_errors(idx), std_errors(idx));
end
