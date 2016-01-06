clc;
clear all;
close all;
format short g;
% ------------------------------------------------------------------------------------------------------------------------------ %
fontsize = 20;
linewidth = 3;
% ------------------------------------------------------------------------------------------------------------------------------ %
% \ddot{x} + \dot{x} + \dot{x} = sin(2t)
N = 199; % Number of sample points in time
t = linspace(0, 2*pi, N+1)';  % Time
omega = 2 * pi / (t(end) - t(1)); % delta f
t = t(1:end-1);
f = sin(2*t); % Forcing functions

F = round(fft(f));

Omega = omega * [0,-1:-1:floor(-N/2),floor(N/2-1):-1:1]';

X = F ./ (1 + i * Omega - Omega.^2); % Solution in frequency domain
x = ifft(X);

xAnalytical = -0.23077 * sin(2*t) -0.15385 * cos(2*t);
figure,
plot(t, x, 'k', ...
       t , xAnalytical, 'r--',...
       'linewidth',linewidth)
xlabel('Time')
ylabel('Displacement')
legend('FFT Solution', 'Analytical Solution')
set(gca,'fontsize',fontsize)