clc;
clear all;
close all;
format short g;
% ------------------------------------------------------------------------------------------------------------------------------ %
fontsize = 20;
linewidth = 3;
% ------------------------------------------------------------------------------------------------------------------------------ %
% \ddot{x} + \dot{x} = sin(2t) * cos(3t)
N = 19; % Number of sample points in time
t = linspace(0, 2*pi, N+1)';  % Time
omega = 2 * pi / (t(end) - t(1)); % delta f
t = t(1:end-1);
f = sin(2*t) + cos(3*t); % Forcing functions

F = round(fft(f));

Omega = omega * [0,-1:-1:floor(-N/2),floor(N/2-1):-1:1]' + eps;

X = F ./ (1 - Omega.^2); % Solution in frequency domain
x = ifft(X);
xAnalytical = -1/3 * sin(2 * t) - 1/8 * cos(3*t);
figure,
subplot(2,1,1)
plot(t, x, 'k', ...
       t , xAnalytical, 'r--',...
       'linewidth',linewidth)
xlabel('Time')
ylabel('Displacement')
legend('FFT Solution', 'Analytical Solution')
set(gca,'fontsize',fontsize)
subplot(2,1,2)
plot(t, 100 * abs(x - xAnalytical)./abs(xAnalytical), 'k',...
       'linewidth',linewidth)
xlabel('Time')
ylabel('Percent Error')
set(gca,'fontsize',fontsize)