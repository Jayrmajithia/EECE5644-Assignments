%% Initialization
xtrain = exam4q1_generateData(1000);
xtest = exam4q1_generateData(10000);
model = 10;
K = 10;
%% MLP
Ntrain = size(xtrain,2);
Ntest = size(xtest,2);
nx = 1;
ny = 1;
paramsTruemodel = cell(model,1);
mseError = zeros(model,1);
ytrue = xtrain(2,:);
xtrain = xtrain(1,:);
y = xtest(2,:);
xtest = xtest(1,:);
for nPerceptrons = 1:model
    sizeParams = [nx; nPerceptrons; ny];

    dummy = ceil(linspace(1,Ntrain,K+1));
    indpartition = zeros(K,2);
    for a = 1:K
        indpartition(a,:) = [dummy(a), dummy(a+1)];
    end
    avgmseError = 0;
    for a = 1:K
        indvalidate = [indpartition(a,1):indpartition(a,2)];
        xvalidate = xtrain(:, indvalidate);
        yvalidate = ytrue(indvalidate);
        if a==1
            indtrain = [indpartition(a,2)+1:Ntrain];
        elseif a==K
            indtrain = [1:indpartition(a,1)-1];
        else
            indtrain = [indpartition(a-1,2)+1:indpartition(a+1,1)-1];
        end
        xt = xtrain(:, indtrain);
        yt = ytrue(indtrain);
        params = fitfunction(xt, yt, sizeParams);
        yvpred= mlpmodel(xvalidate, params);
        avgmseError = avgmseError + calcError(yvalidate, yvpred);
    end
    mseError(nPerceptrons,1) = avgmseError/K;
end
[~,m] = min(mseError);
sizeParams = [nx; m; ny];
params = fitfunction(xtrain, ytrue, sizeParams);
h = mlpmodel(xtest, params);
disp(calcError(y, h));
figure(1), 
plot(xtest, y, '.g');hold on
plot(xtest, h, '.r');
xlabel('X_1'), ylabel('X_2'), legend("True X_2", "Predicated X_2"),
hold off;

%% Functions

function params = fitfunction(x, y, sizeParams)
    options1 = optimset('MaxFunEvals',5000, 'MaxIter', 1000);
%     options2 = optimset('MaxIter', 1000);
    nx = sizeParams(1);
    nPerceptrons = sizeParams(2);
    ny = sizeParams(3);
    params.A = zeros(nPerceptrons, nx);
    params.b = zeros(nPerceptrons,1);
    params.C = zeros(ny, nPerceptrons);
    params.d = mean(y,2);
    vecparamsInit = [params.A(:);params.b;params.C(:);params.d];

    vecParams = fminsearch(@(vecParams)(objectiveFunction(x,y,sizeParams,vecParams)),vecparamsInit, options1);
    params.A = reshape(vecParams(1:nx*nPerceptrons),nPerceptrons,nx);
    params.b = vecParams(nx*nPerceptrons+1:(nx+1)*nPerceptrons);
    params.C = reshape(vecParams((nx+1)*nPerceptrons+1:(nx+1+ny)*nPerceptrons),ny,nPerceptrons);
    params.d = vecParams((nx+1+ny)*nPerceptrons+1:(nx+1+ny)*nPerceptrons+ny);
end

function error = calcError(ytrue, ypred)
    error = mean((ytrue-ypred).^2);
end

function Value = objectiveFunction(x,y,sizeParams,vecParams)
    N = size(x,2); % number of samples
    nX = sizeParams(1);
    nPerceptrons = sizeParams(2);
    nY = sizeParams(3);
    params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
    params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
    params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
    params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);
    H = mlpmodel(x,params);
    Value = sum(sum((y-H).*(y-H),1),2)/N;
end
function h = mlpmodel(x, params)
    N = size(x,2);
    u = params.A*x + repmat(params.b,1,N);  % u = Ax + b
    z = a1(u); % activation function-1 logistic function
    v = params.C*z + repmat(params.d,1,N);  % v = Cz + d
    h = v; % linear output layer activations
end

function out = a1(in)
%     out = 1./(1 + exp(-in)); % Logistic Function
    out = log(1 + exp(in)); % Relu
end