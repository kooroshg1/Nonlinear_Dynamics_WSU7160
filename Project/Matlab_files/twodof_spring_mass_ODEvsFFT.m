function twodof_spring_mass_ODEvsFFT();
clear all
close all
clc
%% ODE45 for 2DOF spring mass system
x0 = [1.5 0 3 0];
[t,xval] = ode45('twodof_spring_mass',[0 2*pi],x0);
plot(t,xval(:,1),t,xval(:,2))
hold on
function [xp] = twodof_spring_mass(t,x);
m1=1; k1=1; m2=1; k2=1;
xp(1) = x(2);
xp(2) = -((k1+k2)/m1)*x(1)-(k2/m1)*x(3);
xp(3) = x(4);
xp(4) = (k2/m2)*x(1) - (k2/m2)*x(3);
xp = [xp(1);xp(2);xp(3);xp(4)];
end
%% FFT
N = 19;
t = linspace(0,2*pi,N+1);
t=t(1:end-1);
x0 = [1.57*ones(N,1) 3.14*ones(N,1)];
% x = fminsearch(@R_twodof_spring_mass,x0)
x = fminsearch(@R_twodof_spring_mass,x0)
plot(t,x(:,1),t,x(:,2))

legend('ODE45 oscillating response', 'Harmonic Balance')

function [R] = R_twodof_spring_mass(x);
m1=1; k1=1; m2=1; k2=1;
N1 =length(x);
T = 2*pi;     % delta_t
time = linspace(0,T,N1+1);
omega = 2*pi/(time(end)-time(1)); % delta f
% t = t(1:end-1)';
Omega = omega*[0, -1:-1:floor(-N1/2), floor(N1/2-1):-1:1]'+eps;

X1 = fft(x(:,1));
dx1 = ifft(1i*Omega.*X1);
ddx1 = ifft(-Omega.^2.*X1);

X2 = fft(x(:,2));
dx2 = ifft(1i*Omega.*X2);
ddx2 = ifft(-Omega.^2.*X2);

R(:,1) = m1.*ddx1+(k1+k2)*x(:,1)-k2*x(:,1)
R(:,2) = m2.*ddx2-k2*x(:,1)+k2*x(:,2)

R = R(:,1)+R(:,2);
R = sum(R.^2);
end
end