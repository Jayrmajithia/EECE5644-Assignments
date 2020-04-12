close all; clear all;
%% Initialize
n = 10;
noise_sigma = (5*0.6)^2;
gammas = linspace(-3,3,21)';
%% Solving it
w = zeros(121,1);
y = zeros(121,1);
costfunc = @(s,g,y,x,w) sum(y-x*w)/(2*s) + (w'*w)/(2*g);
wtrue = unifrnd(-1,1,4,1);
[msqerror, avgmsqerror] = deal(zeros(121, length(gammas)));
for i=1:200
    x = unifrnd(-1, 1, 1, n);
    xm = [x.^3;x.^2;x;ones(1,n)]';
    noise = noise_sigma*randn(n,1);
    ytrue{1,i} = xm*wtrue + noise;
    for j=1:length(gammas)
        g = 10^gammas(j,:);
        initial_w = zeros(4,1);
        wmap{1,i}(:,j) = fminsearch(@(t) costfunc(noise_sigma, g, ytrue{1,i}, xm, t), initial_w);
        ymap{1,i}(:,j)  = xm*wmap{1,i}(:,j) + noise;
    end
    msqerror(i,:) = n\sum((ymap{1,i}-repmat(ytrue{1,i},1,length(gammas))).^2.1);
    avgmsqerror(i,1:length(gammas)) = length(wtrue)\sum((wmap{1,i} - repmat(wtrue, 1, length(gammas))).^2);
end
prctlmsqerror = prctile(avgmsqerror,[0,5,25,50,75,100], 1);
figure(1),
plot(gca,gammas,prctlmsqerror,'linewidth',1);hold on
set(gca, 'YScale', 'log');
xlabel("gammas"); ylabel("Average Mean Squared Error");
legend("Minimum", "25th percentile", "median", "75th Percentile", "Maximum");