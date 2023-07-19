%% BO-MAMP
% If you use our codes, please review our paper. Thank you.
% L. Liu, S. Huang and B. M. Kurkoski, "Memory AMP," in IEEE Transactions on Information Theory, 2022, doi: 10.1109/TIT.2022.3186166.
%
% Problem model: y = Ax + n
% y is the observed vector
% A is a known matrix
% x is the vector to be estimated, where E(x)=0, Var(x)=1
% n is a Gaussian noise vector

%% Parameter Initialization
clc; clear; 
%close all;
rng('default')
    
iter_O = 20;                        % maximum number of iterations for OAMP
iter_M = 70;                        % maximum number of iterations for BO-MAMP
sim_times = 50;
kappa = 100;                         
N = 16384;                          % size of x
beta = 0.5;                         % ratio M / N
M = round(beta * N);                % size of y
L = 3;                              % length of damping 
SNR_dB = 30;                        % SNR(dB)
% distribution of x
P = 0.1;
u_g = 0;                    
v_g = 1 / P;                  
v_x = (P - P^2) * u_g + P * v_g;    % v_x = 1    
u_n = zeros(M, 1);
v_n = v_x ./ (10.^(0.1.*SNR_dB));   
% dia is the vector with singular values of A
T = min(M, N);
dia = kappa.^(-[0:T-1]' / T);
dia = sqrt(N) * dia / norm(dia);    % tr{AA^H} = N, if M > N, replace N with M
% MSE for OAMP / BO-MAMP
MSE_O = zeros(1, iter_O);
MSE_M = zeros(1, iter_M);

%% Simulations
for r = 1 : sim_times
    r
    % source
    b = binornd(1, P, N, 1);
    g = normrnd(u_g , sqrt(v_g), [N, 1]);
    x = b .* g;                          
    % noise
    n = normrnd(u_n, sqrt(v_n), [M, 1]); 
    % A = U*S*V^{H}
    % Let V^{H} be dct, U be I, and V^{H} be fft                                          
    index_ev = randperm(N);
    index_ev = index_ev(1:T);
    index_ev = index_ev';
    x_f = dct(x);
    y = [dia .* x_f(index_ev); zeros(M-N, 1)] + n;
    % LMMSE-OAMP
    [MSE_r, ~] = OAMP(x, y, dia, index_ev, P, u_g, v_g, v_n, iter_O);
    MSE_O = MSE_O + MSE_r;
    % BO-MAMP
    [MSE_M_r, ~] = MAMP(x, y, dia, index_ev, P, u_g, v_g, v_n, L, iter_M);
    MSE_M = MSE_M + MSE_M_r;
end
MSE_O = MSE_O / sim_times;
MSE_M = MSE_M / sim_times;

%% plot figures
plot_len = max([iter_O, iter_M]);
% BO-MAMP
semilogy(0:plot_len, [v_x MSE_M MSE_M(end)*ones(1,plot_len-iter_M)], 'r-', 'LineWidth', 1.5);
hold on;
% LMMSE-OAMP
semilogy(0:plot_len, [v_x MSE_O MSE_O(end)*ones(1,plot_len-iter_O)], 'b-', 'LineWidth', 1.5);
title(['[MAMP] kappa=', num2str(kappa), ';M=', num2str(M), ';N=', num2str(N), ';SNR(dB)=', num2str(SNR_dB)]);
legend('BO-MAMP', 'OAMP/VAMP');
xlabel('Number of iterations', 'FontSize', 11);
ylabel('MSE', 'FontSize', 11);