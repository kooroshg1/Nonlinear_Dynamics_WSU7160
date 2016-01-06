function [R] = residual_DP(theta)
m1=2; l1=1; m2=1; l2=2; g=32.2;

N =length(theta);
T = 2*pi;     % delta_t
t = linspace(0,T,N+1);
omega = 2*pi/(t(end)-t(1)); % delta f
% t = t(1:end-1)';
Omega = omega*[0, -1:-1:floor(-N/2), floor(N/2-1):-1:1]'+eps;

Theta1 = fft(theta(1));
Theta2 = fft(theta(2));
dtheta1 = ifft(1i*Omega.*Theta1);
dtheta2 = ifft(1i*Omega.*Theta2);

ddtheta1 = ifft(-Omega.^2.*Theta1);
ddtheta2 = ifft(-Omega.^2.*Theta2);

a = (m1+m2)*l1;
b = m2*l2*cos(theta(1)-theta(2));
c =  m2*l1*cos(theta(1)-theta(2));
d = m2*l2;
e = m2*l2*dtheta2.^2.*sin(theta(1)-theta(2))+g*(m1+m2).*sin(theta(1));
f =   -m2*l1*dtheta1.^2.*sin(theta(1)-theta(2))+m2*g.*sin(theta(2));

R(:,1) = a.*ddtheta1 + b.*ddtheta2 + e ;
R(:,2) = c.*ddtheta1+ d.*ddtheta2  + f;

end