function robotvisXR(q,th,L,varargin)
FK = FKXR(th,L);
clf
p = axes(q);
if isempty(varargin)
    plot(p,FK(:,1),FK(:,2),'o-');
else
    Xhist = varargin{1};
    hold on
    plot(p,FK(:,1),FK(:,2),'o-')
    plot(p,Xhist(:,1),Xhist(:,2),'-');
    hold off    
end
axis([-sum(L)-1,sum(L)+1,-1,sum(L) + 1])
end
