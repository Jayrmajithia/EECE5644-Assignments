function plot_class(x, label)
    plot(x(1, label==0), x(2, label==0), 'ob');hold on
    plot(x(1, label==1), x(2, label==1), '+r');axis equal,
    legend('Class 0', 'Class 1'); xlabel('x_1'); ylabel('x_2');
end