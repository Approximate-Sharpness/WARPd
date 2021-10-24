function [U, V, y, err_iterations] = WARPd_mc(Omega, epsilon, b, U0, V0, y0, delta, n_iter, k_iter, options)
    
    [Omega,Ik]=sort(Omega); % speeds up one of the projections
    b=b(Ik);
    n1=size(U0,1);    n2=size(V0,1);
    [I1,I2] = ind2sub([n1,n2],Omega);
    addpath(genpath('PROPACK_m'))

    % set default parameters if these are not given
    if ~isfield(options,'store')
        options.store=0;
    end
    
    if ~isfield(options,'type')
        options.type=1;
    end
    
    if ~isfield(options,'upsilon')
        options.upsilon=exp(-1);
    end
    
    if ~isfield(options,'tau')
        options.tau=1;
    end
    
    if ~isfield(options,'display')
        options.display=1;
    end
    
    if ~isfield(options,'L_A')
        options.L_A=1;
    end
    
    if isfield(options,'RG')
        rank_guess = options.RG;
    else
        rank_guess = 5;
    end
    
    if ~isfield(options,'tol')
        options.tol=10^(-20);
    end
    
    if ~isfield(options,'maxiter')
        options.maxiter=10^4;
    end

    % rescale everything
    SCALE=norm(b(:),2);
    U0=U0/SCALE;
    b=b/SCALE;
    y0=y0/SCALE;
    epsilon=epsilon/SCALE;
    delta=delta/SCALE;
    options.SCALE=SCALE;
    
    
    global Uk Vk M
    Uk=U0;    Vk=V0;
    Vk=Vk';
    M=sparse(I1,I2,b,n1,n2);
    [U0,S0,V0] = lansvd('Axz','Atxz',n1,n2,rank_guess,'L',[]);
    U0=U0*S0;
    Vk=Vk';
   
    U = U0;  V = V0; y = y0;
    eps = options.C2*norm(b(:),2);
    err_iterations = [];
    
%     perform the inner iterations
%     fprintf('Performing the inner iterations...\n');
    for j = 1:n_iter
        if options.display==1
            fprintf('n=%d Progress: ',j);
        end
%         beta=options.C1*(delta+eps)/(options.C2*k_iter);
        beta=options.C1*(delta+eps)/(options.C2*ceil(2*options.C1*options.C2*options.L_A/(options.tau*options.upsilon)));
        al = 1/(beta*k_iter); 
        if al>10^12
            al=1;
            options.L_A=1;
        end
        if options.type==5
            options.type=4;
            al=1;
        end
        [U, V, y, err_inner_it,rank_guess] = InnerIt_matcomp(al*b, sqrt(al)*U, sqrt(al)*V, I1, I2, k_iter, options.tau/options.L_A, options.tau/options.L_A, al*epsilon, al, y, options,rank_guess,(j-1)*k_iter);
        U = U/sqrt(al); V = V/sqrt(al);
        if isfield(options,'errFcn')
            err_iterations = [err_iterations(:);
                                err_inner_it(:)];
            if min(err_iterations) < options.tol
                break
            end
        end
        eps = options.upsilon*(delta + eps);
        
        if j*k_iter >= options.maxiter
            break
        end
    end
    U=U*SCALE;
    y=y*SCALE;
    rmpath(genpath('PROPACK_m'))
end


function [Uk, Vk, yk, err_iterations,rank_guess]  = InnerIt_matcomp(b, U0, V0, I1, I2, k_iter, tau1, tau2, epsilon, al, y0, options,rank_guess,JJJ)
    yk = y0; % initiate
    
    if mod(options.type,2)==0
        xk = U0*(V0');
        x_sum = zeros(size(xk));
        y_sum = zeros(size(yk));
    end
    
    if isfield(options,'errFcn')
        err_iterations = zeros(k_iter,1);
    else
        err_iterations = [];
    end
    
    if options.display==1
        pf = parfor_progress(k_iter);
        pfcleanup = onCleanup(@() delete(pf));
    end

    global Uk Vk M
    Uk=U0;    Vk=V0;
    n1=size(U0,1);
    n2=size(V0,1);
    col = [0; find(diff(I2)); length(I1)];
    
    for k = 1:k_iter
        M=sparse(I1,I2,- tau1*yk,n1,n2);
        Vk=Vk';
        [Ukk,Skk,Vkk] = lansvd('Axz','Atxz',n1,n2,rank_guess,'L',[]);
        Vk=Vk';
        
        Skk=diag(Skk);
        if min(Skk)>tau1
            rank_guess = rank_guess+1;
        elseif length(Skk)>1 && Skk(end-1)>tau1
            rank_guess = rank_guess-1;
        end
        rank_guess = min(rank_guess,round(length(b)/(3*(n1+n2))));
        rank_guess = min(rank_guess,80);
        Skk=diag(max(zeros(size(Skk)),(1-tau1./(abs(Skk)+10^(-50))).*Skk));
        Ukk=Ukk*Skk;

        ykk = yk + tau2*(2*UVtOmega(Ukk,Vkk,I1,I2,col)-UVtOmega(Uk,Vk,I1,I2,col)) - tau2*b;
%         ykk = yk + tau2*(UVtOmega(Ukk,Vkk,I1,I2,col)) - tau2*b;
        ykk = prox_gamma( ykk ,tau2*epsilon);
        
        if mod(options.type,2)==0
            x_sum = x_sum + Ukk*(Vkk');
            y_sum = y_sum + ykk;
        end
        
        if isfield(options,'errFcn')
            if mod(options.type,2)==0
                [U,S,V] = lansvd(x_sum,rank_guess,'L',[]);
                err_iterations(k) = options.errFcn(U,options.SCALE*S/(al*k),V);
            else
                err_iterations(k) = options.errFcn(Ukk,options.SCALE*speye(size(Ukk,2))/al,Vkk);
            end
        end
        Uk = Ukk;   Vk = Vkk;
        yk = ykk;
        if options.display==1
            parfor_progress(pf);
        end
        if isfield(options,'errFcn')
            if err_iterations(k)<options.tol
                break
            end
        end
        
        if k+JJJ >= options.maxiter
            break
        end

    end
    
    if mod(options.type,2)==0
        yk = y_sum/k_iter;
        [Uk,Sk,Vk] = lansvd(x_sum/k_iter,rank_guess,'L',[]);
        Uk=Uk*Sk;
    end
    
end

function y_out = prox_gamma(y,rho)    
    n_y = norm(y(:),2) + 1e-43;
    y_out = max(0,1-rho/n_y)*y;
end





