clear;
close all;

% type the filename of data to convert
filename = 'SynData_exp1';

% set the path of the data file
DataPath = fullfile('COMSOL', [filename, '.txt']);

% read the data file from COMSOL
a = readtable(DataPath);

% remove the description lines at the top of the data file
a(1:7, :) = [];

% extract each variable from the data file
x = a.Var1;
y = a.Var2;
u = a.Var3;
v = a.Var4;
h = a.Var5;
mu = a.Var6;

% define the equally-spaced grids in each dimension
x0 = linspace(0,max(x),401);
y0 = linspace(0,max(y),301);
[xd, yd] = meshgrid(x0, y0);

% define another equally-spaced grids in each dimension
x1 = linspace(0,max(x),301);
y1 = linspace(0,max(y),201);
[xd_h, yd_h] = meshgrid(x1, y1);

% interpolate the data in terms of the given grids
ud = griddata(x,y,u,xd,yd);
vd = griddata(x,y,v,xd,yd);
mud = griddata(x,y,mu,xd,yd);

% interpolate the thickness data on another grids
hd = griddata(x,y,h,xd_h,yd_h);

% plot the interpolated data before saving
figure; surf(xd,yd,ud);
shading interp;
colormap jet

% set the positions for the calving front
xct0 = xd(:, end-3:end);
yct0 = yd(:, end-3:end);
xct = reshape(xct0.', [numel(xct0), 1]);
yct = reshape(yct0.', [numel(yct0), 1]);
% set the associated normal vector to the calving front
nnct = horzcat(ones(size(xct)), zeros(size(xct)));

% save the synthetic data for the PINN code of viscosity inversion
save([filename, '.mat'],'xd','yd','ud','vd','xd_h','yd_h','hd', ...
    'mud', "xct", "yct", "nnct");
