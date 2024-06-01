function robotvis3D(qf,T,varargin)
clf
pl = axes(qf);

hold on
p = zeros(4,length(T));
p(4,:) = 1;
for i = 1:length(T)
    p(:,i+1) = T{i}*p(:,1);    
end

if isempty(varargin)
    plot3(pl,p(1,:),p(2,:),p(3,:),'-o');
else
    Xhist = varargin{1};
    hold on
    plot3(pl,p(1,:),p(2,:),p(3,:),'-o');
    plot3(pl,Xhist(1,:),Xhist(1,:),Xhist(3,:),'-');
    hold off    
end

view(45,45)
axis([-3 3 -3 3 -3 3])
end
