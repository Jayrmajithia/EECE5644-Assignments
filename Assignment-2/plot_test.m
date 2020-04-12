function error = plot_test(label, decision, nc, p, x)
    i00 = find(decision==0 & label==0);
    i10 = find(decision==1 & label==0); p10 = length(i10)/nc(1);
    i01 = find(decision==0 & label==1); p01 = length(i01)/nc(2);
    i11 = find(decision==1 & label==1);
    error = (p10*p(1) + p01*p(2))*100;

    plot(x(1, i00),x(2, i00), 'og');hold on
    plot(x(1, i10),x(2, i10), 'or');hold on
    plot(x(1, i01),x(2, i01), '+r');hold on
    plot(x(1, i11),x(2, i11), '+g');hold on
    % plot(px1,px2); hold on
    % axis([px1(1), px1(2), min(x(:,2))-2, max(x(:,2))+2]),
    legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions');
end