clear;
% close all;

% type the filename of data to convert
filename = 'SynData_exp1';

% set the path of the data file
DataPath = [filename, '.txt'];

% read the data file from COMSOL
a = readtable(DataPath);

% get the first variable in the table
x = a.Var1;
% find the first non-Nan value of x
idx = find(~isnan(x),1,'first');
% remove the description lines at the top of the data file
a(1:idx-1,:) = [];

% extract each variable from the data file (1D array)
x = a.Var1;
y = a.Var2;
u = a.Var3;
v = a.Var4;
h = a.Var5;
mu = a.Var6;
eb1 = a.Var7;
eb2 = a.Var8;

% set the shape of the 2D matrix
% (consistent with the grid numbers set in the COMSOL->Export->data)
Shape2D = [400, 250];

% convert the 1D-array data to 2D matrix
xd = reshape(x, Shape2D).';
yd = reshape(y, Shape2D).';
ud = reshape(u, Shape2D).';
vd = reshape(v, Shape2D).';
hd = reshape(h, Shape2D).';
mud = reshape(mu, Shape2D).';
eb1d = reshape(eb1, Shape2D).';
eb2d = reshape(eb2, Shape2D).';

xd_h = xd;
yd_h = yd;

% plot the interpolated data before saving
figure; surf(xd,yd,ud);
shading interp;
colormap jet

% set the positions for the calving front
xct0 = xd(:, end-1:end);
yct0 = yd(:, end-1:end);
xct = reshape(xct0.', [numel(xct0), 1]);
yct = reshape(yct0.', [numel(yct0), 1]);
% set the associated normal vector to the calving front
nnct = horzcat(ones(size(xct)), zeros(size(xct)));

% save the synthetic data for the PINN code of viscosity inversion
save([filename, '.mat'],'xd','yd','ud','vd','xd_h','yd_h','hd', ...
    'mud', "xct", "yct", "nnct");
