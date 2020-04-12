function[x] = genrate_x(label, mu, sigma, Nc, N, n)
    x = zeros(n, N);
    for L = 0:1
        x(:, label==L) = mvnrnd(mu(:, L+1), sigma(:, :, L+1), Nc(L+1))';
    end
end