function[fp,tp, minperror, mini] = solveRoc(discriminant, label, Nc, p, epsilion)
    sortedDiscriminant = sort(discriminant);
    l = length(sortedDiscriminant);
    tou = zeros(l+2);
    tou(1) = sortedDiscriminant(1)-epsilion;
    for i = 1:l-1
        tou(i+1) = (sortedDiscriminant(i) + sortedDiscriminant(i+1))/2  ;
    end
    tou(l+1) = sortedDiscriminant(l)+epsilion;
    fp = zeros(length(tou));
    tp = zeros(length(tou));
    minperror = 1;
    mini = 1;
    for i=1:length(tou)
        decision = (discriminant >= tou(i));
        i10 = find(decision==1 & label==0); fp(i) = length(i10)/Nc(1);
        i11 = find(decision==1 & label==1); tp(i) = length(i11)/Nc(2);
        a = fp(i)*p(1) + p(2) - p(2)*tp(i);
        if a < minperror
            minperror = a;
            mini = i;
        end
    end
end