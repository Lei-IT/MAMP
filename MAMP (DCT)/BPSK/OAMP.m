%% OAMP
function [MSE, Var] = OAMP(x, y, dia, index_ev, v_n, it)
    MSE = zeros(1, it);
    Var = zeros(1, it);
    M = length(y);
    N = length(x);
    u_nle = zeros(N, 1);                    % E(x) = 0
    v_nle = 1;                              % Var(x) = 1
    thres_0 = 1e-10;
    % iterations
    for t = 1 : it
        % LE
        [u_le_p, v_le_p] = LE_OAMP(u_nle, v_nle, dia, index_ev, y, v_n, M, N);
        [u_le, v_le] = Orth(u_le_p, v_le_p, u_nle, v_nle);
        % NLE
        [u_nle_p, v_nle_p] = Demodulation(u_le, v_le, N);
        if v_nle_p <= thres_0
            tmp = (u_nle_p - x)' * (u_nle_p - x) / N;
            MSE(t:end) = max(tmp, thres_0);
            Var(t:end) = thres_0;
            break
        end
        MSE(t) = (u_nle_p - x)' * (u_nle_p - x) / N;                % MSE
        Var(t) = v_nle_p;
        if t == it
            break
        end
        [u_nle, v_nle] = Orth(u_nle_p, v_nle_p, u_le, v_le);
    end
end

%% Orthogonalization
function [u_orth, v_orth] = Orth(u_post, v_post, u_pri, v_pri)
    v_orth = 1 / (1 / v_post - 1 / v_pri);
    u_orth = v_orth * (u_post / v_post - u_pri / v_pri);  
end

%% LE for OAMP
function [u_post, v_post] = LE_OAMP(u, v, dia, index_ev, y, v_n, M, N)
    rho = v_n / v;
    Dia = [dia.^2; zeros(M-N, 1)];                  
    D = 1 ./ (Dia + rho);                           
    Au = A_times_x(u, index_ev, dia, M, N);
    tmp = y - Au;
    tmp = D .* tmp;
    tmp = AH_times_x(tmp, index_ev, dia, M, N);
    u_post = u + tmp;
    v_post = v - v / N * sum(Dia.*D);
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