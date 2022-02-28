function [x_final, y_final, all_iterations, err_iterations] = WARPdSR_reweight(opA, opD, b, x0, y0, delta, n_iter, k_iter, options, weights, N)
    % add the matrix B if supplied to form joint matrix
    if isfield(options,'opB')
        q=options.q;
        opK = @(x,mode) LINFUN(opA,options.opB,q,x,mode);
        b=[b(:);zeros(q,1)];
    else
        q=0;
        opK=opA;
    end
    
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
    
    if ~isfield(options,'lambda')
        options.lambda=1./options.C2;
    end
    
    if ~isfield(options,'L_A')
        fprintf('Computing the norm of A... ');
        l=rand(length(x0),1);
        l=l/norm(l);
        options.L_A = 1;
        for j=1:10 % perform power iterations
            l2=opA(opA(l,1),0);
            options.L_A=1.01*sqrt(norm(l2));
            l=l2/norm(l2);
        end
        fprintf('upper bound is %d\n',options.L_A);
    end
    
    if ~isfield(options,'L_B')
        fprintf('Computing the norm of B... ');
        l=rand(length(x0),1);
        l=l/norm(l);
        options.L_A = 1;
        for j=1:10 % perform power iterations
            l2=options.opB(options.opB(l,1),0);
            options.L_B=1.01*sqrt(norm(l2));
            l=l2/norm(l2);
        end
        fprintf('upper bound is %d\n',options.L_A);
    end

    psi = x0;
    y = y0;
    omega = (norm(b(:),2)/options.lambda)/20;
    all_iterations = cell([n_iter,1]);
    err_iterations = [];
    
    % perform the inner iterations
    fprintf('Performing the inner iterations...\n');
    for j = 1:n_iter
        fprintf('n=%d Progress: ',j);
        tau1=options.tau*options.C1*(delta+omega)/(options.L_A+options.L_B*sqrt(options.lambda^2));
        tau2=options.tau/(options.L_A*options.C1*(delta+omega));
        tau3=options.tau*sqrt(options.lambda^2)/(options.L_B*options.C1*(delta+omega));
        [psi, y, cell_inner_it, err_inner_it, weights] = InnerIt(b, psi, opK, opA, options.opB, k_iter, tau1, tau2, tau3, opD, y, options, q, weights, N);

        for jj=1:length(cell_inner_it)
            cell_inner_it{jj}=cell_inner_it{jj};
        end
        all_iterations{j} = cell_inner_it;
        if isfield(options,'errFcn')
            err_iterations = [err_iterations(:);
                                err_inner_it(:)];
        end
        omega = options.nu*(delta + omega);
    end

    x_final = psi;
    y_final = y;

end


function [x_out, y_out, all_iterations, err_iterations,weights]  = InnerIt(b, x0, opK, opA, opB, k_iter, tau1, tau2, tau3, opD, y0, options, q, weights, N)

    xk = x0;
    yk = y0;
    x_sum = zeros(size(xk));
    y_sum = zeros(size(yk));
    all_iterations = cell([k_iter,1]);
    err_iterations = [];
    if isfield(options,'errFcn')
        err_iterations = zeros(k_iter,1);
    end
    
    if options.display==1
        pf = parfor_progress(k_iter);
        pfcleanup = onCleanup(@() delete(pf));
    end

    if options.shearlets==1
        proxJa = @(x,rho) opD.times(max(zeros(size(x,1),1), abs(x)-rho*weights(:)).*x./(abs(x)+10^(-60)));
            proxJ = @(x,rho) [proxJa(opD.adj(x(1:N^2)),rho);
                            x(N^2+1:end)];
    else
        proxJ =@(x,rho) x;
    end
               
    for k = 1:k_iter
        xkk = proxJ(xk - tau1*opK(yk, 0), tau1*options.lambda);
        z = [tau2*opA(2*xkk - xk, 1); tau3*opB(2*xkk - xk, 1)];
        ykk = prox_dual( yk + z - tau2*b ,options.lambda, q);

        x_sum = x_sum + xkk;
        y_sum = y_sum + ykk;

        if  options.store==1
            if mod(options.type,2)==0
                all_iterations{k} = x_sum/(k);
            else
                all_iterations{k} = xkk;
            end
        end
        
        if isfield(options,'errFcn')
            if mod(options.type,2)==0
                err_iterations(k) = options.errFcn(x_sum/(k));
            else
                err_iterations(k) = options.errFcn(xkk);
            end
        end

        xk = xkk;
        yk = ykk;
        
        if options.display==1
            parfor_progress(pf);
        end

    end
    if options.type==1
        x_out = xk;
        y_out = yk;
    elseif options.type==2
        x_out = x_sum/k_iter;
        y_out = yk;
    elseif options.type==3
        x_out = xk;
        y_out = y_sum/k_iter;
    else
        x_out = x_sum/k_iter;
        y_out = y_sum/k_iter;
    end
    
    if options.reweight==1
        weights = (1./max(abs(opD.adj(x_out(1:N^2))),10^(-4)));
        for j=1:length(weights)/N^2
            Id=(1:N^2)+(j-1)*N^2;
            Id=Id(:);
            weights(Id)=weights(Id)*mean(1./weights(Id));
        end
    end
    

end

function y_out = prox_dual(y,rho,q)
    y=y(:);
    if q==0
        n_y = norm(y(:),2) + 1e-43;
        y_out = min(1,1/n_y)*y;
    else
        n_y = norm(y(1:(end-q)),2) + 1e-43;
        y_out1 = min(1,1/n_y)*y(1:(end-q));
        y_out2 = min(ones(q,1),rho./(abs(y((end-q+1):end))+1e-43)).*y((end-q+1):end);
        y_out=[y_out1; y_out2];
    end
end

function y_out = LINFUN(opA,opB,q,x,mode)
    if mode==1
        y_out=[opA(x,1); opB(x,1)];
    else
        y_out=opA(x(1:(end-q)),0)+opB(x((end-q+1):end),0);
    end
end




