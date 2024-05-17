%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% this is a demo of using varius numerical Jacobian based methods for robot control of a simple 2D system
% In the following example I use a x-R robot where x is the number of
% joints
%% 2R system
L = [2,2];
th0 = [.3,-.3];
FK = FKXR(th0,L);
X = FK(end,:);
Xf = [-1, -.5];
%% 4R system
L = [1,1,1,1];
th0 = [.3,.6,-.3,.5].';
FK = FKXR(th0,L);
X = FK(end,:);
Xf = [-2, 0];
%% its doctor octopus! 
L = .3*ones(1,20);
th0 = ones(1,20).'/10;
FK = FKXR(th0,L);
X = FK(end,:);
Xf = [-2, 2];
%% applying a Jacobian transpose method 

al = .05;
th = th0;
Xhist = [];
q = figure();
q.Position(3) = 600;
q.Position(4) = 400;
axis([-sum(L)-1,sum(L)+1,-1,sum(L) + 1])

for i = 1:1000
    J = JacobianXR(th,L);
    dth = al/norm(J)*J.'*(Xf-X).';
    th = th + dth;
    FK = FKXR(th,L);
    X = FK(end,:);
    Xhist = [Xhist;X];    
    robotvisXR(q,th,L,Xhist);    % visualizing 
%     plot(FK(:,1),FK(:,2),'o-',Xhist(:,1),Xhist(:,2),'-')
%     axis([-sum(L)-1,sum(L)+1,-1,sum(L) + 1])
    pause(.1)
end
%% applying a Pseudo Inverse method 
al = 1;
th = th0;
Xhist = [];
q = figure();
q.Position(3) = 600;
q.Position(4) = 400;
axis([-sum(L)-1,sum(L)+1,-1,sum(L) + 1])

for i = 1:1000
    J = JacobianXR(th,L);
    dth = al*J.'*inv(J*J.')*(Xf-X).'/norm(J);
    th = th + dth;
    FK = FKXR(th,L);
    X = FK(end,:);
    Xhist = [Xhist;X];    
    robotvisXR(q,th,L,Xhist);
    pause(.1)
end

%% using a pseudo inverse
al = 1;
th = th0;
Xhist = [];
q = figure();
q.Position(3) = 600;
q.Position(4) = 400;
axis([-sum(L)-1,sum(L)+1,-1,sum(L) + 1])

for i = 1:1000
    J = JacobianXR(th,L);
    dth = al*pinv(J)*(Xf-X).'/norm(J);
    th = th + dth;
    FK = FKXR(th,L);
    X = FK(end,:);
    Xhist = [Xhist;X];    
    robotvisXR(q,th,L,Xhist);
    pause(.1)
end

%% using a pseudo inverse while keeping endeffector in same direction ( need to work on this )
al = 1;
th = th0;
Xhist = [];
q = figure();
q.Position(3) = 600;
q.Position(4) = 400;
axis([-sum(L)-1,sum(L)+1,-1,sum(L) + 1])

for i = 1:1000
    J = JacobianXR(th,L);
    N = null(J);
    N = N*pinv(N);
    dth = al*pinv(J)*(Xf-X).'/norm(J);
    th = th + dth;
    FK = FKXR(th,L);
    X = FK(end,:);
    Xhist = [Xhist;X];    
    robotvisXR(q,th,L,Xhist);
    pause(.1)
end

%% tracing a heart

t = linspace(0,2*pi*12,1200);

al = .2;
th = th0;
Xhist = [];
q = figure();
q.Position(3) = 600;
q.Position(4) = 400;
axis([-sum(L)-1,sum(L)+1,-1,sum(L) + 1])

for i = 1:1000
    Xf = [sin(t(i)).^3 + 2,(-5*cos(2*t(i))+13*cos(t(i))-2*cos(3*t(i))-cos(4*t(i)))/16+2];    
    J = JacobianXR(th,L);
    dth = al*J.'*inv(J*J.')*(Xf-X).'; %/norm(J);
    th = th + dth;
    FK = FKXR(th,L);
    X = FK(end,:);
    Xhist = [Xhist;X];    
    robotvisXR(q,th,L,Xhist);
    pause(.1)
end

