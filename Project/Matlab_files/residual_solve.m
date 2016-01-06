clear all
close all
clc
<<<<<<< HEAD
N = 5;
t = linspace(0,2*pi,N+1);
t=t(1:end-1);
theta0 = [1.57*ones(N,1) ;3.14*ones(N,1)];
options = optimoptions('fsolve','Display','iter');
options=optimset('MaxIter',1e5,'TolFun',1e-2);
[theta,fval] = fsolve(@residual_DP,theta0,options);


=======
N = 19;
t = linspace(0,2*pi,N+1);
t=t(1:end-1);
theta0 = [1*ones(N,1) 2.5*ones(N,1)];
x0 = 0*ones(N,1);
v0 = 0*ones(N,1);
options = optimoptions('fsolve','Display','iter');
options=optimset('MaxIter',1e5,'TolFun',1e-10);
[theta,fval] = fsolve(@residual_DP,theta0,options);
% [y,fval] = fsolve(@myfun,[x0 v0]);
% [y,fval] = fsolve(@myfun,x0);
>>>>>>> refs/remotes/origin/master

