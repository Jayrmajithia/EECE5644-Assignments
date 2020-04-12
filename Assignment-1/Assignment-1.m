%% P1
mu(:,1) = [-0.1;0]; mu(:,2) = [0.1;0];
sigma(:,:,1) = [1 -0.9;-0.9 1]; sigma(:,:,2) = [1 0.9;0.9 1];
p = [0.8, 0.2];
n = 2; N = 10000;
% Taking the samples 
label = rand(1,N)>=p(1);
Nc = [length(find(label==0)), length(find(label==1))];
x = zeros(n,N);
for l=0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end
% Plotting the original data
figure(1),
plot(x(1,label==0),x(2,label==0),'o');hold on
plot(x(1,label==1),x(2,label==1),'+');axis equal,
legend('Class 0', 'Class 1'),
xlabel('x1'),ylabel('x2')
hold off

% Calculating the discriminants/scores 
discriminant = log(evalGaussian(x,mu(:,2),sigma(:,:,2))) - log(evalGaussian(x,mu(:,1),sigma(:,:,1)));

% Calculating the ROC curve using the SolveRoc designed by me
[fp, tp, minperror, mini] = solveRoc(discriminant,label,Nc,p,0.1);
% Calculating the ROC curve using the ROC curve method availabla in the data
[fpfc, tpfc,t,auc, optimalroc] = perfcurve(label,discriminant,1);
fsp = fp*p(1);
sp = (1-tp)*p(2);
fspc = fpfc*p(1);
spfc = (1-tpfc)*p(2);
xpfc = fspc + spfc;
[minpfc,j]=min(xpfc);
% Displays the minimum error in when the variances are different
disp(minperror);
disp(minpfc);

% Displaying the ROC curve plot using my method results and Matlab predeveloped methods
figure(2),
subplot(2,1,1),
plot(fp,tp,'k-','linewidth',3);hold on
plot(fsp,sp,'r-','linewidth',3);hold on
plot([fsp(mini) fsp(mini)], [0 1], '-b', 'linewidth', 2);hold on
legend('Roc Curve','Probability Error curve','minimum error estimator'),
xlabel("False Possitive"), ylabel("True Possitive"),
hold off
subplot(2,1,2),
plot(fpfc,tpfc,'-k','linewidth',3);hold on
plot(fspc,spfc,'-r','linewidth',3);hold on
plot([fspc(j) fspc(j)], [0 1],'-b', 'linewidth',2);hold on
legend('Roc curve','Probability Error curve','Minimum Probability Error'),
xlabel("False Possitive"), ylabel("True Possitive")
hold off

% Part 2 of question 1
sigma(:,:,1) = [1 0;0 1]; sigma(:,:,2) = [1 0;0 1];
sample_x = zeros(n,N);
for l=0:1
    sample_x(:,label==l) = mvnrnd(mu(:,l+1),sigma(:,:,l+1),Nc(l+1))';
end

% Plotting the data with the unit sigma variance matrix
figure(3),
plot(sample_x(1,label==0),sample_x(2,label==0),'o');hold on
plot(sample_x(1,label==1),sample_x(2,label==1),'+');axis equal,
legend('Class 0', 'Class 1'),
xlabel('x1'),ylabel('x2')
hold off

% Calculating the the discriminant with the Naive Bayes data
discriminantnb = log(evalGaussian(sample_x,mu(:,2),sigma(:,:,2))) - log(evalGaussian(sample_x,mu(:,1),sigma(:,:,1)));

% Calculatingthe ROC curve using my self designed method
[fpnb, tpnb, minperrornb, mininb] = solveRoc(discriminantnb,label,Nc,p,0.7);
% Calculating the ROC curve using Matlab method
[fpnbfc, tpnbfc] = perfcurve(label,discriminantnb,1);
fspnb = fpnb*p(1);
spnb = (1-tpnb)*p(2);
fspnbpc = fpnbfc*p(1);
spnbfc = (1-tpnbfc)*p(2);
xpnbfc = fspnbpc + spnbfc;
[minpnbfc,jnb]=min(xpnbfc);
% The minimum error when Naive Bayesian classifier.
disp(minperrornb);
disp(minpnbfc);

% Plotting the ROC curve and comparing the results using my method and Matlab's method
figure(4),
subplot(2,1,1),
plot(fpnb,tpnb,'k-','linewidth',3);hold on
plot(fspnb,spnb,'r-','linewidth',3);hold on
plot([fspnb(mininb) fspnb(mininb)],[0 1],'b-','linewidth',2);hold on
legend('Roc Curve','Probability Error curve','minimum error estimator'),
xlabel("False Possitive"), ylabel("True Possitive"),
hold off
subplot(2,1,2),
plot(fpnbfc,tpnbfc,'k-', 'linewidth',3);hold on
plot(fspnbpc,spnbfc, 'r-', 'linewidth',3); hold on
plot([fspnbpc(jnb) fspnbpc(jnb)],[0 1], 'b-','linewidth',2); hold on
legend('Roc curve','Probability Error curve','Minimum Probability Error'),
xlabel("False Possitive"), ylabel("True Possitive"),
hold off

%% Question 2
clear all; close all;
n = 2 ;
N = 1000;

% Class 0 parameters
mu(:,1) = [4;0]; mu(:,2) = [6;4];
sigma(:,:,1) = [12 5;5 12]/4; sigma(:,:,2) = [10 -1;-1 4]/5;
p0 = [0.7 0.3];% class priors for class 0

% Class 1 parameters
mu(:,3) = [0;6]; mu(:,4) = [2;7];
sigma(:,:,3) = [10 -1;-1 4]/3; sigma(:,:,4) = [12 -6;-6 12]/6;
p1 = [0.3 0.7];% class priors for class 1

p = [0.6 0.4];
label = rand(1,N) >= p(1);
Nc = [length(find(label == 0)), length(find(label == 1))];
x = zeros(n,N);

% Draw samples from each class total pdf
for i = 1:N
    if label(i) == 0
        distr = rand(1,1) > p0(1);
        if distr == 0
            x(:,i) = mvnrnd(mu(:,1), sigma(:,:,1), 1)';
        else
            x(:,i) = mvnrnd(mu(:,2), sigma(:,:,2), 1)';
        end
    end
    if label(i) == 1
        distr = rand(1,1) > p1(1);
        if distr == 0
            x(:,i) = mvnrnd(mu(:,3), sigma(:,:,3), 1)';
        else
            x(:,i) = mvnrnd(mu(:,4), sigma(:,:,4), 1)';
        end
    end
end

% plotting the classs labels
figure(1), clf,
plot(x(1, label==0), x(2, label==0), 'o'), hold on,
plot(x(1, label==1), x(2, label==1), '+'),
legend('class 0', 'class 1'),
xlabel('x1'), ylabel('x2')

% Cost Matrix
lambda = [0 1;1 0];
gamma = (lambda(2,1) - lambda(1,1))/(lambda(1,2) - lambda(2,2)) * (p(1)/p(2));

class0pdf = p0(1)*evalGaussian(x,mu(:,1), sigma(:,:,1)) + p0(2)*evalGaussian(x,mu(:,2), sigma(:,:,2));
class1pdf = p1(1)*evalGaussian(x,mu(:,3), sigma(:,:,3)) + p1(2)*evalGaussian(x,mu(:,4), sigma(:,:,4));
discriminant = log(class1pdf) - log(class0pdf) - log(gamma);
decision = (discriminant >= 0);

ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1);
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1);
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2);
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2);
p_error = [p01,p10] * Nc'/N;
disp(p_error)

% plotting contors
figure(2),
plot(x(1,ind00), x(2,ind00), 'og');hold on,
plot(x(1,ind10), x(2,ind10), 'or');hold on,
plot(x(1,ind01), x(2,ind01), '+r');hold on,
plot(x(1,ind11), x(2,ind11), '+g');hold on,
axis equal,

hgrid = linspace(floor(min(x(1,:))), ceil(max(x(1,:))), 101);
vgrid = linspace(floor(min(x(2,:))), ceil(max(x(2,:))), 91);
[h,v] = meshgrid(hgrid, vgrid);
% Calculating Discriminant Score Grid Value
dsgv = log(evalGaussian([h(:)';v(:)'],mu(:,3),sigma(:,:,3)) + evalGaussian([h(:)';v(:)'],mu(:,4),sigma(:,:,4))) - log(evalGaussian([h(:)';v(:)'],mu(:,1),sigma(:,:,1)) + evalGaussian([h(:)';v(:)'],mu(:,2),sigma(:,:,2))) - log(gamma);
mindsgv = min(dsgv); maxdsgv = max(dsgv);
discriminantgrid = reshape(dsgv, 91, 101);
figure(2), contour3(hgrid, vgrid, discriminantgrid, [mindsgv * [0.9, 0.6, 0.3], 0, [0.3, 0.6, 0.9] * maxdsgv]);
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ,'Location','southeast'), 
title('Data and their classifier decisions versus true labels'),
xlabel("x1"), ylabel("x2")
hold off