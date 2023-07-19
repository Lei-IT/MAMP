%% MLE for MAMP
% --------------------------------------------------------------------
% log_theta_(i) = log(theta(i)) [theta(i) of the current iteration]
% log_theta_p(i) = log(theta(i)) [theta(i) of the previous iteration]
% theta_w(i) = theta(t-i) * w(i) [i = 1,...,t-1]
% theta_w(t) = theta(1) * w(t)
% theta_w(i) = theta(1) * theta(2t-i) * w(i) [i = t+1,...,2t-1]
% --------------------------------------------------------------------
function [log_theta_, theta_w_, r_hat, r, v_gamma] = MLE_MAMP(x_phi, v_phi, log_theta_, theta_w_, ...
            z, r_hat, B, sign, log_B, index_ev, dia, w_0, w_bar_00, lambda_s, t, v_n, M, N)
    p_bar = zeros(1, t);
    theta = 1 / (lambda_s + v_n / v_phi(t, t));
    if t > 1
        log_theta_p = log_theta_(1:t-1);
        log_theta_(1:t-1) = log_theta_p(1:t-1) + log(theta);
        theta_w_(1:t-2) = theta_w_(1:t-2) .* exp(fliplr(log_theta_(2:t-1) - log_theta_p(1:t-2)));
        theta_w_(t-1) = theta_w(lambda_s, B, sign, log_B, log_theta_(1), t-1, N);
        theta_w_(t) = theta_w(lambda_s, B, sign, log_B, log_theta_(1), t, N);
        log_theta_11 = log_theta_(1) + log_theta_(1);
        theta_w_(2*t-1) = theta_w(lambda_s, B, sign, log_B, log_theta_11, 2*t-1, N);
        if t > 2
            log_theta_12 = log_theta_(1) + log_theta_(2);
            theta_w_(2*t-2) = theta_w(lambda_s, B, sign, log_B, log_theta_12, 2*t-2, N);
            if t > 3
                theta_w_(t+1:2*t-3) = theta_w_(t+1:2*t-3) .* ...
                    exp(log(theta) + fliplr(log_theta_(3:t-1) - log_theta_p(1:t-3)));
            end
        end
        p_bar(1:t-1) = fliplr(theta_w_(1:t-1));
        [c0, c1, c2, c3] = Get_c(p_bar, v_phi, log_theta_, theta_w_, v_n, w_0, w_bar_00, lambda_s, t);
        tmp = c1 * c0 + c2;
        if tmp ~= 0
            xi = (c2 * c0 + c3) / tmp;
        else
            xi = 1;
        end
    else
        [c0, c2, c3] = deal(0);
        c1 = v_n * w_0 + v_phi(1, 1) * w_bar_00;
        xi = 1;
    end
    log_theta_(t) = log(xi);
    p_bar(t) = xi * w_0;
    epsilon = (xi + c0) * w_0;
    v_gamma = (c1 * xi^2 - 2 * c2 * xi + c3) / epsilon^2;
    % r_hat and r
    AHr_ = AH_times_x(r_hat, index_ev, dia, M, N);
    AAHr_ = A_times_x(AHr_, index_ev, dia, M, N);
    r_hat = xi * z(:, t) + theta * (lambda_s * r_hat - AAHr_);
    AHr_ = AH_times_x(r_hat, index_ev, dia, M, N);
    temp = 0;
    for i = 1 : t
        temp = temp + p_bar(i) * x_phi(:, i);
    end
    r = 1 / epsilon * (AHr_ + temp);
end

%% Calculate c0, c1, c2, and c3
function [c0, c1, c2, c3] = Get_c(p_bar, v_phi, log_theta_, theta_w_, v_n, w_0, w_bar_00, lambda_s, t)
    % c0
    c0 = sum(p_bar(1:t-1)) / w_0;
    % c1
    c1 = v_n * w_0 + v_phi(t, t) * w_bar_00;
    % c2
    term_1 = p_bar(1:t-1);
    temp = real(v_phi(t, 1:t-1));
    coeff_1 = v_n + temp * (lambda_s - w_0);
    term_2 = zeros(1, t-1);
    term_2(1) = theta_w_(t);
    term_2(2:t-1) = p_bar(1:t-2) .* exp(log_theta_(2:t-1) - log_theta_(1:t-2));
    c2 = sum(temp .* term_2 - coeff_1 .* term_1);
    % c3
    c3 = 0;
    for i = 1 : t-1
        for j = 1 : t-1
            if 2*t-i-j < t 
                coffe_1 = exp(log_theta_(i) + log_theta_(j) - log_theta_(i+j-t));
            elseif 2*t-i-j == t
                coffe_1 = exp(log_theta_(i) + log_theta_(j) - log_theta_(1));
            else
                coffe_1 = exp(log_theta_(i) + log_theta_(j) - log_theta_(1) - log_theta_(i+j));
            end
            term_1 = (v_n + v_phi(i, j) * lambda_s) * coffe_1 * theta_w_(2*t-i-j);
            if 2*t-i-j+1 < t 
                coffe_2 = exp(log_theta_(i) + log_theta_(j) - log_theta_(i+j-t-1));
            elseif 2*t-i-j+1 == t
                coffe_2 = exp(log_theta_(i) + log_theta_(j) - log_theta_(1));
            else
                coffe_2 = exp(log_theta_(i) + log_theta_(j) - log_theta_(1) - log_theta_(i+j-1));
            end
            term_2 = v_phi(i, j) * coffe_2 * theta_w_(2*t-i-j+1);
            term_3 = v_phi(i, j) * p_bar(i) * p_bar(j);
            c3 = c3 + term_1 - term_2 - term_3;
        end
    end
    c3 = real(c3);
end

%% Ax
function Ax = A_times_x(x, index_ev, dia, M, N)
    x_f = dct(x);
    Ax = [dia .* x_f(index_ev); zeros(M-N, 1)];
end

%% AHx
function AHx = AH_times_x(x, index_ev, dia, M, N)
    tmp = zeros(N, 1);
    T = min(M, N);
    tmp(index_ev) = dia .* x(1:T);
    AHx = idct(tmp);
end

%% theta_(i) * w(j)
function res = theta_w(lambda_s, B, sign, log_B, log_theta_i, j, N)
    tmp = (lambda_s - B) .* sign.^j .* exp(log_theta_i + j * log_B);
    res = 1 / N * sum(tmp); 
end
