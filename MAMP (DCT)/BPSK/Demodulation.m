%% y = x + n, x /in {1, -1} with p_x(1) = p_1
function [u_post, v_post] = Demodulation(u, v, N)
    p1 = 0.5;
    thres = 1e-10;
    if v < thres
        u_post = u;
        v_post = 0;
        return
    end
    u_post = zeros(N, 1);
    v_post = 0;
    for i = 1 : N
        p_1 = p1 / (p1 + (1 - p1) * exp(-2 * u(i) / v));
        u_post(i) = 2*p_1 - 1;
        v_post = v_post + (1 - u_post(i)^2);
    end
    v_post = v_post / N;
end

