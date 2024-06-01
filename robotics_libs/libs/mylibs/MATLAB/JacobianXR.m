function J = JacobianXR(th,L)
J = zeros(2,length(th));
for i = 1:length(th)
    for ii = i:length(th)
        J(1,i) = L(ii)*cos(sum(th(1:ii))) + J(1,i);
        J(2,i) = -L(ii)*sin(sum(th(1:ii))) + J(2,i);
    end
end
end