function [x_final, y_final, all_iterations, err_iterations] = WARPdSR(opA, proxJ, b, x0, y0, delta, n_iter, k_iter, options)
% INPUTS
% ---------------------------
% opA (function handle)     - The sampling operator. opA(x,1) is the forward transform, and opA(y,0) is the adjoint.
% proxJ (function handle)   - The proximal operator of J.
% b (vector)                - Measurement vector.
% x0 (vector)               - Initial guess of x.
% y0 (vector)               - Initial guess of dual vector.
% delta (double)            - Algorithm parameter.
% n_iter (int)              - Number of outer iterations.
% k_iter (int)              - Number of inner iterations.
% options                   - Additional options:
%                               .store tells the algorithm whether to store all the iterations
%                               .type is the type of output (1 is non-ergodic for primal and dual,
%                               2 is ergodic for primal and non-ergodic for dual,
%                               3 is non-ergodic for primal and ergodic for dual,
%                               4 is ergodic for primal and dual, 5 is plain PD iterations).
%                               .C1 and .C2 are constants in the inequality in the paper.
%                               .nu is the algorithmic parameter (optimal is exp(-1))
%                               .L_A is an upper bound on the norm of A.
%                               .tau is the proximal step size
%                               .display = 1 displays progress of each call to InnerIt, 0 surpresses this output
%                               .errFcn is an error function computed at each iteration
%                               .opB operator B for l1 analysis term, this also needs op.q (dim of range of op.B)
%
% OUTPUTS
% -------------------------
% x_final (vector)          - Reconstructed vector (primal).
% y_final (vector)          - Reconstructed vector (dual).
% all_iterations (cell)     - If options.store = 1, this is a cell array with all the iterates, otherwise it is an empty cell array
% err_iterations            - If options.errFcn is given, this is a cell array with all the error function computed for the iterates, otherwise it is an empty cell array
    
    
    % add the matrix B if supplied to form joint matrix
    if isfield(options,'opB')
        q=options.q;
        opB = options.opB;
        b=[b(:);zeros(q,1)];
        if ~isfield(options,'L_B')
            fprintf('Computing the norm of B...');
            l=rand(length(x0),1);
            l=l/norm(l);
            options.L_B = 1;
            for j=1:10 % perform power iterations
                l2=opB(opB(l,1),0);
                options.L_B=1.01*sqrt(norm(l2));
                l=l2/norm(l2);
            end
            fprintf('upper bound for ||B|| is %d\n',options.L_B);
        end
    else
        q=0;
        options.L_B=0;
        opB=[];
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
        fprintf('upper bound for ||A|| is %d\n',options.L_A);
    end

    psi = x0;
    y = y0;
    omega = options.C2*norm(b(:),2);%+norm(x0(:).*weights(:),1);
    all_iterations = cell([n_iter,1]);
    err_iterations = [];
    
    % perform the inner iterations
    fprintf('Performing the inner iterations...\n');
    for j = 1:n_iter
        fprintf('n=%d Progress: ',j);
        if q>0
            tau1=options.tau*options.C1*(delta+omega)/(options.L_A+options.L_B*sqrt(q/options.C2^2));
            tau2=options.tau/(options.L_A*options.C1*(delta+omega));
            tau3=options.tau*sqrt(q/options.C2^2)/(options.L_B*options.C1*(delta+omega));
        else
            tau1=options.tau*options.C1*(delta+omega)/(options.L_A);
            tau2=options.tau/(options.L_A*options.C1*(delta+omega));
            tau3=0;
        end
        [psi, y, cell_inner_it, err_inner_it] = InnerItSQ(b, psi, k_iter, tau1, tau2, tau3, proxJ, opA, opB, y, options, q);
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


function [x_out, y_out, all_iterations, err_iterations]  = InnerItSQ(b, x0, k_iter, tau1, tau2, tau3, proxJ, opA, opB, y0, options, q)
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

    for k = 1:k_iter
        
        if q==0
            xkk = proxJ(xk - tau1*opA(yk, 0), tau1*options.lambda);
            ykk = prox_dual( yk + tau2*opA(2*xkk - xk, 1) - tau2*b ,options.lambda, q);
        else
            z = xk - tau1*opA(yk(1:(end-q)),0) - tau1*opB(yk((end-q+1):end),0);
            xkk = proxJ(z, tau1*options.lambda);
            z = [tau2*opA(2*xkk - xk, 1); tau3*opB(2*xkk - xk , 1)];
            ykk = prox_dual( yk + z - tau2*b ,options.lambda, q);
        end

%         xkk = proxJ(xk - tau1*opK(yk, 0), tau1*options.lambda);
%         ykk = prox_dual( yk + tau2*opK(2*xkk - xk , 1) - tau2*b ,options.lambda, q);

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





