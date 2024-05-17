function X = FKXRP(typ,th,L)
    X = zeros(length(th),2);
    for i = 1:length(th)
        if typ(i) == 's'
            X(i+1,1) = X(i,1) + L(i)*sin(sum(th(1:i)));
            X(i+1,2) = X(i,2) + L(i)*cos(sum(th(1:i)));
        elseif typ(s) == 'p'
            X(i+1,2) = X(i,2) + th(i);
        else
            error('specify jpint types: s for spehrical and p for prismatic\n');
        end
    end
end