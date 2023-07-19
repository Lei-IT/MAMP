%% BO-MAMP 
% theta_w(i) = theta(t-i) * w(i) [i = 1,...,t-1]
% theta_w(t) = theta(1) * w(t)
% theta_w(i) = theta(1) * theta(2t-i) * w(i) [i = t+1,...,2t-1]
function [MSE, Var] = MAMP(x, y, dia, index_ev, v_n, L, it)
    % Initialization
    M = length(y);
    N = length(x);
    beta = M / N;
    lambda = [dia.^2; zeros(M-N, 1)];                                % eigenvalue of AAH
    lambda_s = 0.5 * (max(lambda) + min(lambda));   
    B = lambda_s * ones(M, 1) - lambda;             % eigenvalue of B
    sign = zeros(M, 1);                             
    sign(B > 0) = 1;
    sign(B < 0) = -1;
    log_B = log(abs(B));
    w_0 = 1 / N * (lambda_s * M - sum(B));          
    w_1 = 1 / N * (lambda_s * sum(B) - sum(B.^2)); 
    w_bar_00 = lambda_s * w_0 - w_1 - w_0 * w_0;
    x_phi = zeros(N, it);
    v_phi = zeros(it, it);
    x_phi(:, 1) = zeros(N, 1);                      % E(x) = 0
    log_theta_ = zeros(1, it);
    theta_w_ = zeros(1, 2*it-1);                
    r_hat = zeros(M, 1);
    z = zeros(M, it);
    z(:, 1) = y;                                    % z1 = y - Ax1, x1 = 0
    v_phi(1, 1) = real(1/N * z(:, 1)' * z(:, 1) - beta * v_n) / w_0;
    MSE = zeros(1, it);
    Var = zeros(1, it);
    thres = 1e-6;
    thres_0 = 1e-10;

    % Iterations
    for t = 1 : it
        % MLE
        [log_theta_, theta_w_, r_hat, r, v_gam] = MLE_MAMP(x_phi, v_phi, log_theta_, theta_w_, ...
            z, r_hat, B, sign, log_B, index_ev, dia, w_0, w_bar_00, lambda_s, t, v_n, M, N);
        % NLE
        [x_hat, v_hat] = Demodulation(r, v_gam, N);
        MSE(t) = (x_hat - x)' * (x_hat - x) / N;
        Var(t) = v_hat;
        if v_hat < thres_0
            MSE(t:end) = thres_0;
            Var(t:end) = thres_0;
            break
        elseif t == it
            break
        elseif t > 2
            % We stop algorithm when it converges to save running time
            % It is possible to revise "thres" or comments these codes 
            if Var(t-1) > Var(t) 
                if Var(t-1) - Var(t) < thres
                    MSE(t+1:end) = MSE(t);
                    Var(t+1:end) = Var(t);
                    break;
                end
            else
                if Var(t-2) - Var(t) < thres
                    MSE(t:end) = MSE(t-1);
                    Var(t:end) = Var(t-1);
                    break
                end
            end
        end
        x_phi(:, t+1) = (x_hat / v_hat - r / v_gam) / (1 / v_hat - 1 / v_gam);
        temp = A_times_x(x_phi(:, t+1), index_ev, dia, M, N);
        z(:, t+1) = y - temp;
        for k = 1 : t+1
            v_phi(t+1, k) = (1 / N * z(:, t+1)' * z(:, k) - beta * v_n) / w_0;
            v_phi(k, t+1) = v_phi(t+1, k)';
        end
        % damping
        [x_phi, v_phi, z] = Damping_NLE(x_phi, v_phi, z, L, t);
    end
end

%% Damping (NLE)
function [x_phi, v_phi, z] = Damping_NLE(x_phi, v_phi, z, L, t)
    l = min(L, t+1);
    v_temp = v_phi(t+2-l:t+1, t+2-l:t+1);
    if rcond(v_temp) < 1e-16 || min(eig(v_temp)) < 0
        if v_phi(t+1, t+1) > v_phi(t, t)
            x_phi(:, t+1) = x_phi(:, t);
            v_phi(t+1, t+1) = v_phi(t, t);
            v_phi(1:t, t+1) = v_phi(1:t, t);
            v_phi(t+1, 1:t) = v_phi(t, 1:t);
            z(:, t+1) = z(:, t);
        end
    else
        temp = (v_temp)^(-1);
        v_s = real(sum(sum(temp)));
        zeta = sum(temp, 2) / v_s;
        v_phi(t+1, t+1) = 1 / v_s;
        x_phi(:, t+1) = sum(zeta.'.*x_phi(:, t+2-l:t+1), 2);
        z(:, t+1) = sum(zeta.'.*z(:, t+2-l:t+1), 2);
        for k = 1 : t
            v_phi(k, t+1) = sum(zeta.'.*v_phi(k, t+2-l:t+1));
            v_phi(t+1, k) = v_phi(k, t+1)';
        end
    end
end

%% Ax
function Ax = A_times_x(x, index_ev, dia, M, N)
    x_f = dct(x);
    Ax = [dia .* x_f(index_ev); zeros(M-N, 1)];
end