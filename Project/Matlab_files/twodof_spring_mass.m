function [xp] = twodof_spring_mass(t,x)
m1=1; k1=1; m2=1; k2=1;
xp(1) = x(2);
xp(2) = -((k1+k2)/m1)*x(1)-(k2/m1)*x(3);
xp(3) = x(4);
xp(4) = (k2/m2)*x(1) - (k2/m2)*x(3);
xp = [xp(1);xp(2);xp(3);xp(4)];
end
