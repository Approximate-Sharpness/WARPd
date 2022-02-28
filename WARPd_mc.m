function [U, V, y, err_iterations] = WARPd_mc(Omega, epsilon, b, U0, V0, y0, delta, n_iter, k_iter, options)

    [Omega,Ik]=sort(Omega); % speeds up one of the projections
    b=b(Ik);
    n1=size(U0,1);    n2=size(V0,1);
    [I1,I2] = ind2sub([n1,n2],Omega);
    addpath(genpath('PROPACK_m'))
    [III,JJJ,~] = find(sparse(I1,I2,abs(b)+1,n1,n2));
    Jcol = compJcol(JJJ);

    % set default parameters if these are not given
    if ~isfield(options,'store')
        options.store=0;
    end
    if ~isfield(options,'type')
        options.type=1;
    end
    if ~isfield(options,'nu')
        options.nu=exp(-1);
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
   
    U = U0;  V = V0; y = y0;
    omega = options.C2*norm(b(:),2);%/20; this can speed things up
    err_iterations = [];
    
%     perform the inner iterations
%     fprintf('Performing the inner iterations...\n');
    rank_guess2=rank_guess;
    for j = 1:n_iter
        if options.display==1
            fprintf('n=%d Progress: ',j);
        end 
        tau1=options.tau*options.C1*(delta+omega)/(options.L_A*options.C2);
        tau2=options.tau*options.C2/(options.L_A*options.C1*(delta+omega));
        [U, V, y, err_inner_it,rank_guess2] = InnerIt_matcomp(b, U, V, I1, I2, k_iter, tau1, tau2, epsilon, y, options,rank_guess2,(j-1)*k_iter,III,Jcol);
        if isfield(options,'errFcn')
            err_iterations = [err_iterations(:);    err_inner_it(:)];
            if min(err_iterations) < options.tol
                break
            end
        end
        if j==1
            rank_guess2=rank_guess;
        end
        omega = options.nu*(delta + omega);
        if j*k_iter >= options.maxiter
            break
        end
    end
    rmpath(genpath('PROPACK_m'))
end


function [Uk, Vk, yk, err_iterations,rank_guess]  =  InnerIt_matcomp(b, U0, V0, I1, I2, k_iter, tau1, tau2, epsilon, y0, options,rank_guess,JJJ,III,Jcol)
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
    XXX=mexProjOmega(Uk,Vk,III,Jcol);
    OP2=[];
    OP2.tol=options.tol/10;
    
    for k = 1:k_iter
        M=sparse(I1,I2,- tau1*yk,n1,n2);
        Vk=Vk';
        [Ukk,Skk,Vkk] = lansvd('Axz','Atxz',n1,n2,rank_guess,'L',[],OP2);
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
        
        XXX2=mexProjOmega(Ukk,Vkk,III,Jcol);
        ykk = yk + tau2*(2*XXX2- 1*XXX)- tau2*b;
        ykk = max(0,1-tau2*epsilon/norm(ykk(:),2))*ykk;
        
        if mod(options.type,2)==0
            x_sum = x_sum + Ukk*(Vkk');
            y_sum = y_sum + ykk;
        end
        
        if isfield(options,'errFcn')
            if mod(options.type,2)==0
                [U,S,V] = lansvd(x_sum,rank_guess,'L',[]);
                err_iterations(k) = options.errFcn(U,S/k,V);
            else
                err_iterations(k) = options.errFcn(Ukk,speye(size(Ukk,2)),Vkk);
            end
        end
        Uk = Ukk;   Vk = Vkk;
        XXX=XXX2;
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






