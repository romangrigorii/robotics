function X = FKXR(th,L)
    X = zeros(length(th),2);
    for i = 1:length(th)
        X(i+1,1) = X(i,1) + L(i)*sin(sum(th(1:i)));
        X(i+1,2) = X(i,2) + L(i)*cos(sum(th(1:i)));
    end
end