%% LQR fun!
% I want to solve an inverted pedulum problem using LQR control design 

%% This is for a block on ice system

A = [[0 1];[0 -.2]];
B = [[0];[1]];
Q = [[1 0];[0 1]]
R = [[.01]];
C = zeros(3,3);
C(1:2,1:2) = Q;
C(3,3) = R;
F = zeros(2,3);
F(1:2,1:2) = A;
F(1:2,3) = B;

V = [[1 0];[0 1]];
v = [[1 0];[0 1]];
for t = 1:1000
    Q = C + F.'*V*F;
    q = F.'*v;
    
    Qxu = 