clear
close all;
%%% This code requires the cilib library at
%%% https://github.com/vegarant/cilib and shearlab at http://shearlab.math.lmu.de/
%% Read the image and set problem parameters
fname_core = 'brain1';
N = 128*4; 
fname = sprintf('%s_%d.png', fname_core, N);
im = double(imread(fname))/255;
im=im(:);

alpha=0.5;
alpha0=0.4;
alpha1=0.2;

%% Set up sampling operator
srate = 0.15; % Subsampling rate
[idx, str_id] = cil_sp2_power_law(N, round(N*N*srate), 2, [N/2,N/2]);%cil_spf2_DAS(N, nbr_samples, 1, 1);
opA = @(x, mode) cil_op_fourier_2d(x, mode, N, idx);

% %% If you want to look at the sampling pattern, run the lines below;
% Z = zeros([N,N]);
% Z(idx) = 1;
% figure; imagesc(Z); colormap('gray');
% axis equal off

%% Sample the image and add noise
b=opA(im(:),1);
b0=b;
noise_v=randn(size(b));
noise_v=noise_v/norm(noise_v(:));
b=b0+noise_v*0.05*norm(b(:)); % add the noise

%% Set up the shearlet analysis operator
sparse_param = [1, 1, 2];
[D,Sys] = getShearletOperator([N,N], sparse_param);
tt=opA(b,0);
weights = 1./max(abs(D.adj(tt(1:N^2))),10^(-4));
for j=1:length(weights)/N^2
    Id=(1:N^2)+(j-1)*N^2;
    Id=Id(:);
    weights(Id)=weights(Id)*max(1./weights(Id));
end
              
% Set up TV
[FgradX,FgradY] = getPeriodicGradForw([N,N]); % get periodic forward gradient
TVmat0=FgradX+1i*FgradY;
options.opB = @(x,mode) Amat_maker(alpha*TVmat0,x,mode);
options.q = N^2;

[BgradX,BgradY] = getPeriodicGradBack([N,N]); % get periodic backward gradient

TVmat1=[[FgradX;    FgradY],-speye(2*N^2,2*N^2)];
TVmat2=[BgradX, 0*BgradX;
        BgradY, BgradX;
        0*BgradX, BgradY];

options.opB = @(x,mode) Amat_maker([alpha1*TVmat1; [sparse(3*N^2,N^2),alpha0*TVmat2]],x,mode);
options.q = 5*N^2;

%% Algorithmic parameters
options.C1=sqrt(1/log(N));
options.C2=sqrt(log(N));
options.lambda=1/options.C2;
options.lambda=1/10000;
epsilon = max(1.2*norm(b(:)-b0(:)),10^(-8));
delta = 10^(-5);

options.tau = 1;
options.upsilon=exp(-1);

k_iter = 5; n_iter = 7;
options.reweight=1;
options.shearlets=1;

%% Run WARPd
x0 = 0*opA(b,0);
y0=0*[b;zeros(options.q,1)];
options.type=4;
options.errFcn=@(x) norm(reshape(min(max(real(x(1:N^2)),0),1), N,N)-reshape(im,N,N),'fro')/norm(im(:));

[rec_BP,y_BP,~,E_BP] = WARPd_reweight(opA, epsilon,D, b, x0, y0, delta, 5, 10, options, weights, N);
[rec_SR,y_SR,~,E_SR] = WARPdSR_reweight(opA, D, b, x0, y0, delta, 5, 10, options, weights, N);

im_BP = reshape(min(max(real(rec_BP(1:N^2)),0),1), N,N);
im_SR = reshape(min(max(real(rec_SR(1:N^2)),0),1), N,N);
im = reshape(im,N,N);

%% compute TV solutions
options.shearlets=0;
k_iter = 50;
n_iter = 5;
[rec_GTV,~,~,E_GTV] = WARPd_reweight(opA, epsilon, D, b, x0, y0, delta, n_iter, k_iter, options, weights, N);
im_GTV = reshape(min(max(real(rec_GTV(1:N^2)),0),1), N,N);
%%
options.opB = @(x,mode) Amat_maker(alpha*TVmat0,x,mode);
options.q = N^2;
opA = @(x, mode) cil_op_fourier_2d2(x, mode, N, idx);
x0 = 0*opA(b,0);
y0=0*[b;zeros(options.q,1)];
[rec_TV,~,~,E_TV] = WARPd_reweight(opA, epsilon, D, b, x0, y0, delta, n_iter, k_iter, options, weights, N);
im_TV = reshape(min(max(real(rec_TV(1:N^2)),0),1), N,N);


%%
im_SR = reshape(min(max(real(rec_SR(1:N^2)),0),1), N,N);
close all
Y=cell(5,1);
Y{1}=im;    Y{2}=im_TV; Y{3}=im_GTV;    Y{4}=im_BP; Y{5}=im_SR;

for j=1:5
    YY=Y{j};
    psnr(im2uint16(YY), im2uint16(im))
    figure
    imagesc(YY); colormap('gray');caxis([0,1]);axis equal;axis off
    axis equal
    a2 = axes();
    a2.Position = [0.2 0.64 0.2 0.3]; % xlocation, ylocation, xsize, ysize
    imagesc(YY((260:350)+10,(270:350)+50))
    a2.XTick=100000;
    a2.YTick=100000;
    set(gca,'XColor','r')
    a2.YAxis.LineWidth=1;
    a2.XAxis.LineWidth=1;
    set(gca,'YColor','r')
    axis equal
end

%%
figure
semilogy(E_BP,'linewidth',2)
hold on
semilogy(E_SR,'linewidth',2)
xlim([0,30])
legend({'WARPd','WARPdSR'},'location','northeast','fontsize',14,'interpreter','latex')

function x_out = Amat_maker(A,x,mode)
    if mode==1
        x_out = A*x;
    else
        x_out = (A')*x;
    end
end

function y = cil_op_fourier_2d(x, mode, N, idx)
    if (~isvector(x))
        error('Input is not a vector');
    end
    R = round(log2(N));
    if (abs(2^R - N) > 0.5) 
        error('Input length is not equal 2^R for some R ∈ ℕ');
    end
    if (mode == 1)
        z = reshape(x(1:N^2),N,N);
        z = fftshift(fft2(z))/N;
        y = z(idx);
        y=y(:);
    else
        z = zeros([N, N]);
        z(idx) = x;
        z = ifft2(ifftshift(z))*N;
        y = z(:);
        y=[y;zeros(2*N^2,1)];
    end
end

function y = cil_op_fourier_2d2(x, mode, N, idx)
    if (~isvector(x))
        error('Input is not a vector');
    end

    R = round(log2(N));

    if (abs(2^R - N) > 0.5) 
        error('Input length is not equal 2^R for some R ∈ ℕ');
    end

    if (mode == 1)
        z = reshape(x(1:N^2),N,N);
        z = fftshift(fft2(z))/N;
        y = z(idx);
        y=y(:);
    else 
        z = zeros([N, N]);
        z(idx) = x;
        z = ifft2(ifftshift(z))*N;
        y = z(:);
    end
end

function [gradX,gradY] = getPeriodicGradForw(m)
per1Dx = spdiags([-1*ones(m(1),1),ones(m(1),1)],[0,1],m(1),m(1));
per1Dy = spdiags([-1*ones(m(2),1),ones(m(2),1)],[0,1],m(2),m(2));

per1Dx(end,1) = 1;
per1Dy(end,1) = 1;

gradX = kron(per1Dx,speye(m(2)));
gradY = kron(speye(m(1)),per1Dy);
end

function [gradX,gradY] = getPeriodicGradBack(m)
per1Dx = spdiags([-1*ones(m(1),1),ones(m(1),1)],[-1,0],m(1),m(1));
per1Dy = spdiags([-1*ones(m(2),1),ones(m(2),1)],[-1,0],m(2),m(2));

per1Dx(1,end) = -1;
per1Dy(1,end) = -1;

gradX = kron(per1Dx,speye(m(2)));
gradY = kron(speye(m(1)),per1Dy);
end

function [D,Sys] = getShearletOperator(m,ndir)

    if length(m) == 2
        if length(ndir) == 1
            Sys         = SLgetShearletSystem2D(0,m(1),m(2),(ndir));
        else
            Sys         = SLgetShearletSystem2D(0,m(1),m(2),length(ndir),ndir);
        end

        D.d         = size(Sys.shearlets);
        D.spectra   = (sum(abs(Sys.shearlets).^2,3));

        array       = @(x,m) reshape(x,m);
        vec         = @(x) x(:);

        D.adj       = @(x) vec(SLsheardec2Dswapped(array(x,m),Sys));
        D.times     = @(x) vec(SLshearrec2Dswapped(array(x,D.d),Sys));


        % swapp elemtns in Sys
        t1              = Sys.RMS(1);
        Sys.RMS(1)      = Sys.RMS(end);
        Sys.RMS(end)    = t1;    
        t2              = Sys.shearletIdxs(1,:);
        Sys.shearletIdxs(1,:) = Sys.shearletIdxs(end,:);
        Sys.shearletIdxs(end,:) = t2;
        D.Sys = Sys;
    elseif length(m) == 3
        if length(ndir) == 1
            Sys         = SLgetShearletSystem3D(0,m(1),m(2),m(3),(ndir));
        else
            Sys         = SLgetShearletSystem3D(0,m(1),m(2),m(3),length(ndir),ndir);
        end
        D.d             = size(Sys.shearlets);
        D.spectra       = sum(abs(Sys.shearlets).^2,4);
        
        array           = @(x,m) reshape(x,m);
        vec             = @(x) x(:);
        
        D.adj           = @(x)  vec(SLsheardec3D(array(x,m),Sys));  
        D.times         = @(y)  vec(SLshearrec3D(array(y,D.d),Sys));
    end
end

function y =  SLsheardec2Dswapped(x,Sys)
   y            = SLsheardec2D(x,Sys);
   temp1        = y(:,:,1);
   y(:,:,1)     = y(:,:,end);
   y(:,:,end)   = temp1;
end

function y =  SLshearrec2Dswapped(x,Sys) 
   temp1        = x(:,:,1);
   x(:,:,1)     = x(:,:,end);
   x(:,:,end)   = temp1;
   y            = SLshearrec2D(x,Sys);
end

