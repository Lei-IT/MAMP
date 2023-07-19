function [u_post, v_post] = Demodulation(u, v, P, u_g, v_g, N)
    EXP_MAX = 50;
    EXP_MIN = -50;
    ug = u_g * ones(N, 1);
    vg = v_g;
    % p1
    a = sqrt((v + vg) / v);
    b = 0.5 * ((u - ug).^2 / (v + vg) - (u.^2) / v);
    % set threshold
    b(b > EXP_MAX) = EXP_MAX;
    b(b < EXP_MIN) = EXP_MIN;
    c = (1 - P) / P;
    p1 = 1 ./ (1 + a * exp(b) * c);
    % Gaussian addition
    v1 = (vg^(-1) + v^(-1))^(-1);
    u1 = v1 * (vg^(-1) * ug + v^(-1) * u);
    % post u and v
    u_post = p1 .* u1;
    v_post = mean(((p1 - p1.^2) .* (u1.^2) + p1 * v1));
end
