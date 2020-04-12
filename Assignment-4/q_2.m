%% Initialize
c = 2;
K = 10;
[xtrain, ytrain] = generateMultiringDataset(c, 1000);
[xtest, ytest] = generateMultiringDataset(c, 10000);
%%
N = size(xtrain,2);
dummy = ceil(linspace(0, N, K+1));
ytrain(ytrain==2) = 0;
ytest(ytest==2) = 0;
xtrain = xtrain'; ytrain = ytrain';
xtest = xtest'; ytest = ytest';
for k=1:K
    indPartitionLimits(k,:) = [dummy(k) + 1, dummy(k+1)];
end
Clist = 10.^linspace(-1, 9, 11); sigmaList = 10.^linspace(-2, 3, 13);
ab = zeros(length(Clist), length(sigmaList));
for s = 1:length(sigmaList)
    sigma = sigmaList(s);
    for c = 1:length(Clist)
        box = Clist(c);
        tot = 0;
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xv = xtrain(indValidate, :); % Using folk k as validation set
            yv = ytrain(indValidate, :);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            end
            % using all other folds as training set
            xt = xtrain(indTrain, :);
            yt = ytrain(indTrain, :);
            svmk = fitcsvm(xt, yt, 'BoxConstraint', c, 'KernelFunction', 'RBF', 'KernelScale', sigma);
            yvpred = svmk.predict(xv);
            tot = tot + calcAccuracy(yv, yvpred);
        end
        ab(c,s) = tot/N;
    end
end
[~,indi] = max(ab(:)); [indBestc, indBests] = ind2sub(size(ab), indi);
Cbest = Clist(indBestc); Sigmabest = sigmaList(indBests);
svm = fitcsvm(xtrain, ytrain, 'BoxConstraint', Cbest, 'KernelFunction', 'RBF', 'KernelScale', Sigmabest);
yprd = svm.predict(xtest);
disp("Showing Accuracy");
disp(calcAccuracy(ytest,yprd));
ind00 = find(ytest==0 & yprd==0); ind01 = find(ytest==0 & yprd==1);
ind11 = find(ytest==1 & yprd==1); ind10 = find(ytest==1 & yprd==0);
crctind = [ind00;ind11];
incrctind = [ind01;ind10];
hold off
confusionchart(ytest, yprd);
figure(2),
contour(log10(Clist), log10(sigmaList), ab', 20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
color = rand(c,3);
figure(3),
plot(xtest(crctind(:,1),1), xtest(crctind(:,1),2),'g.');hold on
plot(xtest(incrctind(:,1),1), xtest(incrctind(:,1),2), 'r.');
xlabel("x_1"), ylabel("x_2"), title("Showing incorrect data"),
legend("True Data", "False Data"),hold off

%% Function
function acc = calcAccuracy(ytrue, ypred)
    cm = confusionmat(ytrue, ypred);
    c = size(cm,1);
    n = size(ytrue,1);
    s = 0;
    for i=1:c
        s = s + cm(i,i);
    end
    acc = s/n;
end