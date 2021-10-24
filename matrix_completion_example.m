%%% Example code for matrix completion with the train station data set %%%
clear
% close all
%% set up the matrix and experiment

load('Train_matrix.mat','DIST');
X_real=round(sqrt(DIST));
% X_real=round((DIST));
n1=size(DIST,1);
n2=n1;
p=0.07;
m=round(n1*n2*p);
maxiter=100;

Omega = randsample(n1*n2,m); % these are the random entries we see
[I1,I2] = ind2sub([n1,n2],Omega);
b=X_real(Omega); % data we see in form of matrix entries

%% set the algorithmic parameters (see code for additional input parameters that can be given)
options.L_A=0.7;%min(1.6*sqrt(m/(n1*n2)),1); % this may need to be adjusted to one for some problems
options.C2=1; options.C1=sqrt(n1*n2/m); % these may need to be increased for some problems
options.tau=1; % proximal step size
options.upsilon=exp(-1); % optimal rescale factor
epsilon = 10^(-10); % this is an "exact" recovery problem
delta = min(options.C2*epsilon,0.0001);

k_iter = ceil(2*options.C1*options.C2*options.L_A/(options.tau*options.upsilon));
n_iter = ceil(maxiter/k_iter);

options.errFcn = @(U,S,V) norm((U*S*V')-X_real,'fro')/norm(X_real,'fro'); % this a test function we want to compute for each iterate

%% execute WARPd
U0 = eye(n1,15); V0 = eye(n2,15); S0 = zeros(15,15);    y0 = 0*b;
options.display=1;
options.type=1; % set to 2 for ergodic
[~, ~, ~, err1] = WARPd_mc(Omega, epsilon, b, U0, V0, y0, delta, n_iter, k_iter, options);

X_real=round((DIST)); % perform experiment with M^(2)
options.L_A=min(1.6*sqrt(m/(n1*n2)),1);
k_iter = ceil(2*options.C1*options.C2*options.L_A/(options.tau*options.upsilon));
n_iter = ceil(maxiter/k_iter);
b=X_real(Omega);
options.errFcn = @(U,S,V) norm((U*S*V')-X_real,'fro')/norm(X_real,'fro'); % this a test function we want to compute for each iterate
[~, ~, ~, err2] = WARPd_mc(Omega, epsilon, b, U0, V0, y0, delta, n_iter, k_iter, options);
%% convergence plot
figure
semilogy(err1,'linewidth',2)
hold on
semilogy(err2,'linewidth',2)
legend
ylim([10^(-5),1])
legend({'$M^{(1)}$','$M^{(2)}$'},'interpreter','latex','fontsize',14,'location','northeast')
xlim([0,100])

%% singular value plot
figure
S1=svd(round(sqrt(DIST)));
S2=svd(round((DIST)));
semilogy(S1/S1(1),'o')
hold on
semilogy(S2/S2(1),'o')
xlim([0,2569])
legend({'$M^{(1)}$','$M^{(2)}$'},'interpreter','latex','fontsize',14,'location','northeast')



    
