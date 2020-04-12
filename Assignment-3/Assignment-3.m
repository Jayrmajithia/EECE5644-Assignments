c = 3; % number of class
[xtrain, ytrain] = generateMultiringDataset(c,1000);
[xtest, ytest] = generateMultiringDataset(c,10000);
M = 7; % Model order
k = 10; % number of k-fold
%% q-1
Ntrain = size(ytrain,2);
Ntest = size(ytest,2);
ytrain1 = zeros(c, Ntrain);
ytest1 = zeros(c, Ntest);
for i=1:Ntrain
    ytrain1(ytrain(i), i) = 1;
end
for i=1:Ntest
    ytest1(ytest(i), i) = 1;
end
classtesterror1 = zeros(1,M);
classtesterror2 = zeros(1,M);
classmodel = cell(M,2);
for model=1:M
    dummy = ceil(linspace(1,Ntrain,k+1));
    indpartition = zeros(k,2);
    for a = 1:k
        indpartition(a,:) = [dummy(a), dummy(a+1)];
    end
    crossnet = cell(k,2);
    classerror1 = zeros(1,k);
    classerror2 = zeros(1,k);
    for a = 1:k
        indvalidate = [indpartition(a,1):indpartition(a,2)];
        xvalidate = xtrain(:, indvalidate);
        yvalidate = ytrain1(:, indvalidate);
        ylabelvalidate = ytrain(indvalidate);
        if a==1
            indtrain = [indpartition(a,2)+1:Ntrain];
        elseif a==k
            indtrain = [1:indpartition(a,1)-1];
        else
            indtrain = [indpartition(a-1,2)+1:indpartition(a+1,1)-1];
        end
        xt = xtrain(:, indtrain);
        yt = ytrain1(:, indtrain);
        net1 = patternnet(model);
        net1.layers{1}.transferFcn = 'logsig';
        net1 = train(net1, xt, yt);
        predy = net1(xvalidate);
        labely = vec2ind(predy);
        crossnet{a,1} = net1;
        classerror1(1,a) = calcError(ylabelvalidate, labely, size(predy,1));
        net2 = patternnet(model);
        net12.layers{1}.transferFcn = 'poslin';
        net2 = train(net2, xt, yt);
        predy = net2(xvalidate);
        labely = vec2ind(predy);
        crossnet{a,2} = net2;
        classerror2(1,a) = calcError(ylabelvalidate, labely, size(predy,1));
    end
    [~, minice] = min(classerror1);
    classmodel{model,1} = crossnet{minice,1};
    m = crossnet{minice,1};
    predy = m(xtrain);
    labely = vec2ind(predy);
    classtesterror1(1,model) = calcError(ytrain, labely, size(predy,1));
    [~, minice] = min(classerror2);
    classmodel{model,2} = crossnet{minice,2};
    m = crossnet{minice,2};
    predy = m(xtrain);
    labely = vec2ind(predy);
    classtesterror2(1,model) = calcError(ytrain, labely, size(predy,1));
end
[a1, ce1] = min(classtesterror1);
[a2, ce2] = min(classtesterror2);
if a1 < a2
    m = classmodel{ce1,1};
    disp("Using Sigmoid function");
else
    m = classmodel{ce2,2};
    disp("Using Relu Function");
end
predy = m(xtest);
labely = vec2ind(predy);
disp("Minimum Classification error");
disp(calcError(ytest, labely, size(predy,1)));
figure(2),
plot(1-classtesterror1, '-b');hold on
plot(1-classtesterror2, '-r'); 
xlabel("Model order");
ylabel("1-classificationerror");
legend("Sigmoid", "Relu");
hold off;
figure(3),
confusionchart(ytest, labely);
title("Classification error using MLP");
%% q-2
model = 7;
gmm = cell(c,1);
options = statset('MaxIter',1000);
for i=1:c
    gmm{i} = fitgmm(xtrain(:,ytrain==i), k, model);
end
xt = xtest';
probytest = zeros(c,1);
for i=1:c
    probytest(i,1) = length(find(ytest==i))/size(ytest,2);
end
predy = zeros(1, size(ytest,2));
for i=1:size(xt,1)
    ab = zeros(c,1);
    for class=1:c
        m = gmm{class};
        p = pdf(m, [xt(i,1) xt(i,2)]);
        ab(class,1) = p * probytest(class,1);
    end
    [~,label] = max(ab);
    predy(:,i) = label;
end
disp("Classification error using EM for GMM");
disp(calcError(ytest, predy, c));
figure(4),
confusionchart(ytest, predy);
title("Classification error using EM for GMM");
%% function 
function ce = calcError(ytrue, ypred, c)
    Ntrain = size(ytrue,2);
    cm = confusionmat(ytrue, ypred);
    sum = 0;
    for i=1:c
        sum = sum + cm(i,i);
    end
    ce = 1-(sum/Ntrain);
end

function gmmodel = fitgmm(x, k, model)
   n = size(x,2);
   options = statset('MaxIter',1000);
   s = zeros(model,1);
   dummy = ceil(linspace(1,n,k+1));
   indpartition = zeros(k,2);
   for a = 1:k
       indpartition(a,:) = [dummy(a), dummy(a+1)];     
   end 
   gmmmodel = cell(model,1);
   s = zeros(model,1);
   for m=1:model
       for a=1:k
            indvalidate = [indpartition(a,1):indpartition(a,2)];
            xvalidate = x(:, indvalidate);
            if a==1
                indtrain = [indpartition(a,2)+1:n];
            elseif a==k
                indtrain = [1:indpartition(a,1)-1];
            else
                indtrain = [1:indpartition(a-1,2)+1, indpartition(a+1,1)-1:n];
            end
            xtrain = x(:, indtrain);
            gl = fitgmdist(xtrain', m, 'Options', options, 'RegularizationValue',0.1);
            s(m,1) = s(m,1) + gl.BIC;
       end
   end
   [~, m] = min(s);
   gmmodel = fitgmdist(x', m, 'Options', options, 'RegularizationValue',0.1);
end