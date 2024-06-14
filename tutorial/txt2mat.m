clear;
close all;

% type the data filename to convert
filename = 'SynData_exp1';

% read the data from COMSOL file
a = readtable([filename, '.txt']);

% remove the description lines at the top of the data file
a(1:7, :) = [];

% extract each variable from the data file
x = a.Var1;
y = a.Var2;
u = a.Var3;
v = a.Var4;
h = a.Var5;
nu = a.Var6;

% define the equally-spaced grids in each dimension
x0 = linspace(0,max(x),301);
y0 = linspace(0,max(y),201);
[xq, yq] = meshgrid(x0, y0);

% interpolate the data in terms of the given grids
uq = griddata(x,y,u,xq,yq);
vq = griddata(x,y,v,xq,yq);
hq = griddata(x,y,h,xq,yq);
mq = griddata(x,y,nu,xq,yq);

% plot the interpolated data before saving
figure; surf(xq,yq,uq);
shading interp;
colormap jet

% save the synthetic data for the PINN code of viscosity inversion
save([filename, '.mat'],'xq','yq','uq','vq','hq','mq');
