% DOA demo: Robust M-estimation (Huber / Tyler / t-Student) vs classical CBF / MVDR / MUSIC
% -----------------------------------------------------------------------------
% In these code, we will present known basline algortihms for DOA
% estimation. we will use CBF, MVDR, MUSIC
%
% Lastly we will compare it to the robust M-estimator - the SBLv5.12,
% utilizng the repository of:
% Robust and Sparse M-Estimation of DOA
% Y. Park, P. Gerstoft, C. F. Mecklenbräuker, E. Ollila
%
% -----------------------------------------------------------------------------
%
% Written by Adam Shusterman - 5.9.2025
%
clear; close all; clc;

%% Array & signal parameters
M   = 12;              % # sensors
fc  = 4e3;             % Center frequency (narrowband)
c   = 343;             % Speed of sound [m/s] (or use EM wave c for RF)
lmb = c / fc;          % Wavelength

d   = 0.5 * lmb;       % Inter-element spacing (half-wavelength)
K   = 128;             % # snapshots
SNR = 0;               % per-source SNR [dB]

% True DOAs (degrees)
true_DOAs = [-60, 25];
D = numel(true_DOAs);  % # sources (for MUSIC)

% Grid for spectra
theta_grid = -90:0.25:90;
rad_grid = deg2rad(theta_grid);

%% Generate data with heavy tails + outliers

A = createSteeringMatrix(M, d, lmb, rad_grid);
A_true = createSteeringMatrix(M, d, lmb, deg2rad(true_DOAs));

rng(0);

S = (randn(D, K) + 1j*randn(D, K)) / sqrt(2);
S = S .* db2mag(SNR); % scale for SNR

% Base noise (complex Gaussian, unit variance)
W = (randn(M, K) + 1j*randn(M, K)) / sqrt(2);

% Add heavy-tailed behavior via t-Student mixing
nu_t = 2.1;
w_t = sqrt(nu_t ./ chi2rnd(nu_t, 1, K));
W = W .* w_t;

X = A_true * S + W;  % Generate the observed data matrix

%% Covariance estimates
R_scm = scm_scatter(X);

R_h = huber_scatter(X, 0.9);

R_t = t_student_scatter(X, 2.1);

R_tyl = tyler_scatter(X);
%% Spectra
% 1) Bartlett / CBF using SCM
P_cbf = bartlett_doa_estimation(A, R_scm);

% 2) MVDR/Capon
P_mvdr_scm = mvdr_doa_estimation(A, R_scm);
P_mvdr_tyl = mvdr_doa_estimation(A, R_tyl);
P_mvdr_t   = mvdr_doa_estimation(A, R_t);

% 3) MUSIC (needs #sources D)
P_music_scm = music_doa_estimation(A, R_scm, D);
P_music_tyl = music_doa_estimation(A, R_tyl, D);
P_music_t   = music_doa_estimation(A, R_t,   D);

%% Plotting results

% Calculate the mean and median power for each spectrum
mean_power_cbf = mean(10*log10(P_cbf));
median_power_cbf = median(10*log10(P_cbf));
diff_cbf = max(10*log10(P_cbf)) - mean_power_cbf;

% CBF Plot with Ground Truth
figure;
plot(theta_grid, 10*log10(P_cbf), 'LineWidth', 3, 'Color', 'b');
hold on;
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
grid on;
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Power (dB)', 'FontSize', 12);
title(['CBF Spectrum with Ground Truth (Peak - Mean: ', num2str(diff_cbf, '%.2f'), ' dB)'], 'FontSize', 14);
legend('CBF Spectrum', 'Location', 'Best');
set(gca, 'FontSize', 12, 'LineWidth', 1.5);

% MVDR Plots in Subplots
figure;
subplot(3,1,1);
mean_power_mvdr_scm = mean(10*log10(P_mvdr_scm));
diff_mvdr_scm = max(10*log10(P_mvdr_scm)) - mean_power_mvdr_scm;
plot(theta_grid, 10*log10(P_mvdr_scm), 'LineWidth', 3, 'Color', 'r');
hold on;
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
grid on;
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Power (dB)', 'FontSize', 12);
title(['MVDR Spectrum (SCM) with Ground Truth (Peak - Mean: ', num2str(diff_mvdr_scm, '%.2f'), ' dB)'], 'FontSize', 14);
legend('MVDR SCM', 'Location', 'Best');
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
ylim([-20 10]); % Set y-limits relevant to MVDR SCM

subplot(3,1,2);
mean_power_mvdr_tyl = mean(10*log10(P_mvdr_tyl));
diff_mvdr_tyl = max(10*log10(P_mvdr_tyl)) - mean_power_mvdr_tyl;
plot(theta_grid, 10*log10(P_mvdr_tyl), 'LineWidth', 3, 'Color', 'g');
hold on;
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
grid on;
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Power (dB)', 'FontSize', 12);
title(['MVDR Spectrum (Tyler) with Ground Truth (Peak - Mean: ', num2str(diff_mvdr_tyl, '%.2f'), ' dB)'], 'FontSize', 14);
legend('MVDR Tyler', 'Location', 'Best');
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
ylim([-20 10]); % Set y-limits relevant to MVDR Tyler

subplot(3,1,3);
mean_power_mvdr_t = mean(10*log10(P_mvdr_t));
diff_mvdr_t = max(10*log10(P_mvdr_t)) - mean_power_mvdr_t;
plot(theta_grid, 10*log10(P_mvdr_t), 'LineWidth', 3, 'Color', 'm');
hold on;
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
grid on;
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Power (dB)', 'FontSize', 12);
title(['MVDR Spectrum (t-Student) with Ground Truth (Peak - Mean: ', num2str(diff_mvdr_t, '%.2f'), ' dB)'], 'FontSize', 14);
legend('MVDR t-Student', 'Location', 'Best');
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
ylim([-20 10]); % Set y-limits relevant to MVDR t-Student

% MUSIC Plots
figure;
subplot(3,1,1);
mean_power_music_scm = mean(10*log10(P_music_scm));
diff_music_scm = max(10*log10(P_music_scm)) - mean_power_music_scm;
plot(theta_grid, 10*log10(P_music_scm), 'LineWidth', 3, 'Color', 'c');
hold on;
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
grid on;
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Power (dB)', 'FontSize', 12);
title(['MUSIC Spectrum (SCM) with Ground Truth (Peak - Mean: ', num2str(diff_music_scm, '%.2f'), ' dB)'], 'FontSize', 14);
legend('MUSIC SCM', 'Location', 'Best');
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
ylim([-20 15]); % Set y-limits relevant to MUSIC SCM

subplot(3,1,2);
mean_power_music_tyl = mean(10*log10(P_music_tyl));
diff_music_tyl = max(10*log10(P_music_tyl)) - mean_power_music_tyl;
plot(theta_grid, 10*log10(P_music_tyl), 'LineWidth', 3, 'Color', 'yellow');
hold on;
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
grid on;
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Power (dB)', 'FontSize', 12);
title(['MUSIC Spectrum (Tyler) with Ground Truth (Peak - Mean: ', num2str(diff_music_tyl, '%.2f'), ' dB)'], 'FontSize', 14);
legend('MUSIC Tyler', 'Location', 'Best');
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
ylim([-20 15]); % Set y-limits relevant to MUSIC Tyler

subplot(3,1,3);
mean_power_music_t = mean(10*log10(P_music_t));
diff_music_t = max(10*log10(P_music_t)) - mean_power_music_t;
plot(theta_grid, 10*log10(P_music_t), 'LineWidth', 3, 'Color', 'm');
hold on;
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
grid on;
xlabel('Angle (degrees)', 'FontSize', 12);
ylabel('Power (dB)', 'FontSize', 12);
title(['MUSIC Spectrum (t-Student) with Ground Truth (Peak - Mean: ', num2str(diff_music_t, '%.2f'), ' dB)'], 'FontSize', 14);
legend('MUSIC t-Student', 'Location', 'Best');
set(gca, 'FontSize', 12, 'LineWidth', 1.5);
ylim([-20 15]); % Set y-limits relevant to MUSIC t-Student

opts = SBLSet;
opts.Nsource = D;
opts.convergenceMu = 1;

% Active-index refinement (same spirit as the demo)
opts.activeIndices = 1;
opts.activeIndRepN = 10;
opts.convergence.min_iteration = opts.activeIndRepN;

% Peak selection params matched to your grid resolution
degRes        = theta_grid(2) - theta_grid(1);
errorDOAsepP  = max(1, floor(5/degRes) - 1);
errorDOApeak  = D + 2;

% Gaussian SBL
[gammaIndG, reportG] = SBL_v5p12(A, X, 'SBL-G', inf, opts, errorDOApeak, errorDOAsepP);

% Student-t SBL
[gammaIndT, reportT] = SBL_v5p12(A, X, 'SBL-T', 2.1, opts, errorDOApeak, errorDOAsepP);

% Huber SBL
[gammaIndH, reportH] = SBL_v5p12(A, X, 'SBL-H', 0.9, opts, errorDOApeak, errorDOAsepP);

% Tyler SBL
[gammaIndTy, reportTy] = SBL_v5p12(A, X, 'SBL-Tyl', M, opts, errorDOApeak, errorDOAsepP);

% Plot SBL spectrum
P_sblG  = reportG.results.final_iteration.gamma;
P_sblT  = reportT.results.final_iteration.gamma;
P_sblH  = reportH.results.final_iteration.gamma;
P_sblTy = reportTy.results.final_iteration.gamma;
% Convert to dB
P_sblG_db  = 10*log10(max(P_sblG,1e-16));
P_sblT_db  = 10*log10(max(P_sblT,1e-16));
P_sblH_db  = 10*log10(max(P_sblH,1e-16));
P_sblTy_db = 10*log10(max(P_sblTy,1e-16));

% Plot all SBL spectra in one figure
figure;
hold on;
plot(theta_grid, P_sblG_db, 'cyan', 'LineWidth', 2);
plot(theta_grid, P_sblT_db, 'magenta', 'LineWidth', 2);
plot(theta_grid, P_sblH_db, 'green', 'LineWidth', 2);
plot(theta_grid, P_sblTy_db, 'blue', 'LineWidth', 2);
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
xlabel('Angle (deg)');
ylabel('Power (dB)');
title('SBL Spectra Comparison');
legend('SBL Gaussian', 'SBL t-Student', 'SBL Huber', 'SBL Tyler', 'Location', 'Best');
grid on;
hold off;

% 5) MVDR/Capon
P_mvdr_scm = mvdr_doa_estimation(A, reportG.results.final_iteration.RY);
P_mvdr_tyl = mvdr_doa_estimation(A, reportTy.results.final_iteration.RY);
P_mvdr_h = mvdr_doa_estimation(A, reportH.results.final_iteration.RY);
P_mvdr_t   = mvdr_doa_estimation(A, reportT.results.final_iteration.RY);

P_sblG_db  = 10*log10(max(P_mvdr_scm,1e-16));
P_sblT_db  = 10*log10(max(P_mvdr_t,1e-16));
P_sblTyl_db  = 10*log10(max(P_mvdr_tyl,1e-16));
P_sblH_db  = 10*log10(max(P_mvdr_h,1e-16));

figure;
hold on;
plot(theta_grid, P_sblG_db, 'cyan', 'LineWidth', 2);
plot(theta_grid, P_sblT_db, 'magenta', 'LineWidth', 2);
plot(theta_grid, P_sblH_db, 'green', 'LineWidth', 2);
plot(theta_grid, P_sblTyl_db, 'green', 'LineWidth', 2);
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
xlabel('Angle (deg)');
ylabel('Power (dB)');
title('MVDR SBL Spectra Comparison');
legend('MVDR SBL Gaussian', 'MVDR SBL t-Student', 'MVDR SBL Huber', 'MVDR SBL Tyler', 'Location', 'Best');
grid on;
hold off;

P_music_scm = music_doa_estimation(A, reportG.results.final_iteration.RY, D);
P_music_tyl = music_doa_estimation(A, reportTy.results.final_iteration.RY, D);
P_music_h = music_doa_estimation(A, reportH.results.final_iteration.RY, D);
P_music_t   = mvdr_doa_estimation(A, reportT.results.final_iteration.RY, D);

P_sblG_db  = 10*log10(max(P_music_scm,1e-16));
P_sblT_db  = 10*log10(max(P_music_t,1e-16));
P_sblTyl_db  = 10*log10(max(P_music_tyl,1e-16));
P_sblH_db  = 10*log10(max(P_music_h,1e-16));

figure;
hold on;
plot(theta_grid, P_sblG_db, 'cyan', 'LineWidth', 2);
plot(theta_grid, P_sblT_db, 'magenta', 'LineWidth', 2);
plot(theta_grid, P_sblH_db, 'green', 'LineWidth', 2);
plot(theta_grid, P_sblTyl_db, 'red', 'LineWidth', 2);
xline(true_DOAs, 'r--', 'Ground Truth', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'middle');
xlabel('Angle (deg)');
ylabel('Power (dB)');
title('MUSIC SBL Spectra Comparison');
legend('MUSIC SBL Gaussian', 'MUSIC SBL t-Student', 'MUSIC SBL Huber', 'MUSIC SBL Tyler', 'Location', 'Best');
grid on;
hold off;

snr_list = -21:3:21;
interactive_gui(snr_list, true_DOAs, theta_grid, A, A_true, K, M)