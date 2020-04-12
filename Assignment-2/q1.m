clear all; close all;
%% Initailize the labels
n = 2;
n1 = 10; n2 = 100; n3 = 1000; nv = 10000;
p = [0.9 0.1];
mu(:, 1) = [-2;0]; sigma(:, :, 1) = [1 -0.9;-0.9 2];
mu(:, 2) = [2;0]; sigma(:, : , 2) = [2 0.9;0.9 1];
l1 = rand(1,n1) >= p(1); l2 = rand(1,n2) >= p(1);
l3 = rand(1,n3) >= p(1); lv = rand(1,nv) >= p(1);
N1 = [length(find(l1==0)),length(find(l1==1))];
N2 = [length(find(l2==0)),length(find(l2==1))];
N3 = [length(find(l3==0)),length(find(l3==1))];
Nv = [length(find(lv==0)),length(find(lv==1))];
x1 = genrate_x(l1, mu, sigma, N1, n1, n);
x2 = genrate_x(l2, mu, sigma, N2, n2, n);
x3 = genrate_x(l3, mu, sigma, N3, n3, n);
xv = genrate_x(lv, mu, sigma, Nv, nv, n);
%% Part 1
figure(1),
plot_class(xv, lv),
title("True dataset with 10000 samples"), hold off

discriminant = log(evalGaussian(xv, mu(:, 2), sigma(:, :, 2)))- log(evalGaussian(xv, mu(:, 1), sigma(:, :, 1)));
[fp, tp] = perfcurve(lv, discriminant, 1);
fsp = fp*p(1);
sp = (1-tp)*p(2);
xp = fsp + sp;
[minperror, i] = min(xp);
minperror = minperror*100;
disp(minperror);

figure(2),
plot(fp, tp, '-k'); hold on
plot(fsp(i), sp(i), 'ob');
xlabel("False Positive"); ylabel("True Positive");
legend("Roc Curve", "Practical Minimum Probability Error")
hold off

lambda = [0 1;1 0];
gamma = log(p(1)) - log(p(2));
decision = discriminant >= gamma;
i00 = find(lv==0 & decision==0); i10 = find(lv==1 & decision==0);
i11 = find(lv==1 & decision==1); i01 = find(lv==0 & decision==1);

% Plotting contors
figure(3),
plot(xv(1,i00), xv(2,i00), 'og');hold on,
plot(xv(1,i10), xv(2,i10), 'or');hold on,
plot(xv(1,i01), xv(2,i01), '+r');hold on,
plot(xv(1,i11), xv(2,i11), '+g');hold on,
axis equal,
hgrid = linspace(floor(min(xv(1,:))), ceil(max(xv(1,:))), 101);
vgrid = linspace(floor(min(xv(2,:))), ceil(max(xv(2,:))), 91);
[h,v] = meshgrid(hgrid, vgrid);
% Calculating Discriminant Score Grid Value
dsgv = log(evalGaussian([h(:)';v(:)'], mu(:, 2), sigma(:, :, 2))) - log(evalGaussian([h(:)';v(:)'], mu(:, 1), sigma(:, :, 1)));
mindsgv = min(dsgv); maxdsgv = max(dsgv);
discriminantgrid = reshape(dsgv, 91, 101);
figure(3), contour3(hgrid, vgrid, discriminantgrid, [mindsgv * [0.9, 0.6, 0.3], 0, [0.3, 0.6, 0.9] * maxdsgv]);
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ,'Location','southeast'), 
title('Data and their classifier decisions versus true labels'),
xlabel("x_1"), ylabel("x_2")
hold off

%% Part 2 % Fitting Logistic regression on sample data
hfunc = @(x,w) 1/1+exp(-w'*x);
cf = @(n, l, x, theta) (-1/n) * sum(l.*hfunc(x,theta) + (1-l).*(1-hfunc(x,theta)));
zx1 = [ones(n1,1), x1']';
zx2 = [ones(n2,1), x2']';
zx3 = [ones(n3,1), x3']';
zxv1 = [ones(nv,1), xv']';
initial_theta = zeros(n+1, 1);
[theta1, cost1] = fminsearch(@(t)(cf(n1, l1, zx1, t)), initial_theta);
[theta2, cost2] = fminsearch(@(t)(cf(n2, l2, zx2, t)), initial_theta);
[theta3, cost3] = fminsearch(@(t)(cf(n3, l3, zx3, t)), initial_theta);
d1 = exp(-theta1'*zx1) >= (p(1)/p(2));
d2 = exp(-theta2'*zx2) >= (p(1)/p(2));
d3 = exp(-theta3'*zx3) >= (p(1)/p(2));
figure(4),
subplot(3,1,1),
e1 = plot_test(l1, d1, N1, p, x1);
title("Train dataset with 10 Samples with probaility error of " + e1 + "%"), hold off
subplot(3,1,2),
e2 = plot_test(l2, d2, N2, p, x2);
title("Train dataset with 100 Samples with probaility error of " + e2 + "%"), hold off
subplot(3,1,3),
e3 = plot_test(l3, d3, N3, p, x3);
title("Train dataset with 1000 Samples with probaility error of " + e3 + "%"), hold off

decision(1,:) = exp(-theta1'*zxv1) >= (p(1)/p(2));
decision(2,:) = exp(-theta2'*zxv1) >= (p(1)/p(2));
decision(3,:) = exp(-theta3'*zxv1) >= (p(1)/p(2));
figure(5),
e1 = plot_test(lv, decision(1,:), Nv, p, xv);
title("Validation dataset with 10 Samples as training dataset with probaility error of " + e1 + "%"), hold off
figure(6),
e2 = plot_test(lv, decision(2,:), Nv, p, xv);
title("Validation dataset with 100 Samples as training dataset with probaility error of " + e2 + "%"), hold off
figure(7),
e3 = plot_test(lv, decision(3,:), Nv, p, xv);
title("Validation dataset with 1000 Samples as training dataset with probaility error of " + e3 + "%"), hold off
%% Part 3 Fitting Logistics Quadratics Equation 
zx4 = [ones(n1,1), x1', x1(1,:)'.*x1(1,:)', x1(1,:)'.*x1(2,:)', x1(2,:)'.*x1(2,:)']';
zx5 = [ones(n2,1), x2', x2(1,:)'.*x2(1,:)', x2(1,:)'.*x2(2,:)', x2(2,:)'.*x2(2,:)']';
zx6 = [ones(n3,1), x3', x3(1,:)'.*x3(1,:)', x3(1,:)'.*x3(2,:)', x3(2,:)'.*x3(2,:)']';
zxv2 = [ones(nv,1), xv', xv(1,:)'.*xv(1,:)', xv(1,:)'.*xv(2,:)', xv(2,:)'.*xv(2,:)']';
initial_theta2 = zeros(6, 1);
[theta4, cost4] = fminsearch(@(t)(cf(n1, l1,zx4, t)), initial_theta2);
[theta5, cost5] = fminsearch(@(t)(cf(n2, l2,zx5, t)), initial_theta2);
[theta6, cost6] = fminsearch(@(t)(cf(n3, l3,zx6, t)), initial_theta2);
d4 = exp(-theta4'*zx4) >= (p(1)/p(2));
d5 = exp(-theta5'*zx5) >= (p(1)/p(2));
d6 = exp(-theta6'*zx6) >= (p(1)/p(2));
figure(8),
subplot(3,1,1),
e4 = plot_test(l1, d4, N1, p, x1);
title("Train dataset with 10 Samples with probaility error of " + e4 + "%"), hold off
subplot(3,1,2),
e5 = plot_test(l2, d5, N2, p, x2);
title("Train dataset with 100 Samples with probaility error of " + e5 + "%"), hold off
subplot(3,1,3),
e6 = plot_test(l3, d6, N3, p, x3);
title("Train dataset with 100 Samples with probaility error of " + e6 + "%"), hold off

decision1(1,:) = exp(-theta4'*zxv2) >= (p(1)/p(2));
decision1(2,:) = exp(-theta5'*zxv2) >= (p(1)/p(2));
decision1(3,:) = exp(-theta6'*zxv2) >= (p(1)/p(2));
figure(9),
e1 = plot_test(lv, decision1(1,:), Nv, p, xv);
title("Validation dataset with 10 Samples as training dataset with probaility error of " + e1 + "%"), hold off
figure(10),
e2 = plot_test(lv, decision1(2,:), Nv, p, xv);
title("Validation dataset with 100 Samples as training dataset with probaility error of " + e2 + "%"), hold off
figure(11),
e3 = plot_test(lv, decision1(3,:), Nv, p, xv);
title("Validation dataset with 1000 Samples as training dataset with probaility error of " + e3 + "%"), hold off