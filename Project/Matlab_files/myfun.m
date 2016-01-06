
function myfun()
clc
close all
clear all

%ddx + x = sin(2*t)
w = 2;
f = w/(2*pi);
N = 9;
T = 1/f;
t = linspace(0, T, N+1);
omega = 2 * pi / (t(end) - t(1));
t = t(1:end-1)';

Omega = omega*[0,-1:-1:floor(-N/2),floor(N/2-1):-1:1]'+eps;

 f = sin(2.*t);

F = fft(f);

% X_H = F./(1-Omega.^2);
% x_h = ifft(X_H);

x0 = 1*ones(N,1);

[X1,fval] = fminsearch(@objfun,x0);

function error = objfun(x)
     X = fft(x);
 ddx = ifft(-Omega.^2.*X);

 RES = (((ddx + x - f)));
 error = sum(abs(RES).^2);
end

 figure(1)
 xAnalytical = -sin(2.*t) / 3;
 size(X1)
% abs([X1 x_h xAnalytical])
%  plot(t,X1,t,xAnalytical, 'ro',t,x_h,'k-');
plot(t,X1,t,xAnalytical, 'ro');
 
 end
