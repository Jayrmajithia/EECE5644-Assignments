clear all; close all
%% Fixing the True values
alpha_true = [0.2 0.3 0.7 0.5];
mu_true = [5 0 2 -5; 0 6 3 -3];
sigma_true(:, :, 1) = [3 1;1 5];
sigma_true(:, :, 2) = [3 -0.9;-0.9 5];
sigma_true(:, :, 3) = [4 1;1 6];
sigma_true(:, :, 4) = [4 -1;-1 6];
M = 100;
%%
averagegmmlikelihood10 = zeros(M,6);
averagegmmlikelihood100 = zeros(M,6);
averagegmmlikelihood1000 = zeros(M,6);
for i=1:M
    x10 = randGMM(10, alpha_true, mu_true, sigma_true)';
    x100 = randGMM(100, alpha_true, mu_true, sigma_true)';
    x1000 = randGMM(1000, alpha_true, mu_true, sigma_true)';
    averagegmmlikelihood10(i,:)= validate(x10);
    averagegmmlikelihood100(i,:)= validate(x100);
    averagegmmlikelihood1000(i,:)= validate(x1000);
end
%% Plotting the data
ar10 = count(M, averagegmmlikelihood10);
ar100 = count(M, averagegmmlikelihood100);
ar1000 = count(M, averagegmmlikelihood1000);
figure(1),
subplot(3,1,1),
bar(ar10(:,1),ar10(:,2));
ylabel("Ocuurence with best fit");
title("Number of best fits happens in a Model with 10 Data Samples")
subplot(3,1,2),
bar(ar100(:,1),ar100(:,2));
ylabel("Ocuurence with best fit");
title("Number of best fits happens in a Model with 100 Data Samples");
subplot(3,1,3),
bar(ar1000(:,1),ar1000(:,2));
title("Number of best fits happens in a Model with 1000 Data Samples")
xlabel("Number of Components");ylabel("Ocuurence with best fit");
%% Functions
function map = count(M,avgmm)
    qr = zeros(6,2);
    for i=1:6
        qr(i,1) = i;
    end
    for i=1:M
        a = zeros(1,6);
        a(1,:) = avgmm(i,:);
        [~,j]= min(a);
        qr(j,2) = qr(j,2) + 1;
    end
    map = qr;
end
function x = validate(D)
    B=12;
    option = statset('MaxIter',200);
    sum = zeros(6,1);
    for i=1:B
        for k=1:6
            try
                gm{k} = fitgmdist(D,k,'Options',option);
                sum(k,1) = sum(k,1) + gm{k}.AIC;
            catch exception
            end
        end
    end
    x = sum./B;
end
function x = randGMM(N,alpha,mu,Sigma)
    d = size(mu,1); % dimensionality of samples
    cum_alpha = [0,cumsum(alpha)];
    u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
    for m = 1:length(alpha)
        ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
        x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    end
end

%%%
function x = randGaussian(N,mu,Sigma)
    % Generates N samples from a Gaussian pdf with mean mu covariance Sigma
    n = length(mu);
    z =  randn(n,N);
    A = Sigma^(1/2);
    x = A*z + repmat(mu,1,N);
end