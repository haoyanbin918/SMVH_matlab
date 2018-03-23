function [W, Z] = SMVH_graddesc_SGD(X, P, d, W_ini)
%%
% This gradient descent function uses the stochastic gradient descent (SGD)
% strategy. Using the standard gradient descent strategy may get better
% performance than using the SGD one. 
%
% [W,Z] = SMVH_graddesc_SGD(X, P, d, W_ini)
%
% The high-dimensional datapoints are specified by X. 
% The target hash code dimensionality is specified in d (default = 100).
% The function returns the embedded points in Z and combination coefficients 
% and bias parameter in W.
% W_ini is for initializing W. 
% This function is part of SMVH Implementation for paper:"Stochastic Multi-view Hashing for
% Large-scale Near-duplicate Video Retrieval"
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Yanbin Hao, Hefei University of Technology
%%
    if ~exist('d', 'var') || isempty(d)
        d = 100;
    end
    
    [n, dim] = size(X);                 % number of instances
    Xone = [X, ones(n,1)];              % add one to X
    
    % We suggest users use W_ini to initialize W, which could accelerate
    % the gradient descent and get better performance.
    % W_ini can be getted by adopting SMVH_graddesc_stan.m with a less
    % training data.
    if ~exist('W_ini', 'var') || isempty(W_ini)
        % Initialize combination coefficients randomly (close to origin),
        % bias b is added in the W (the last row).
        W = 0.00001*rand(dim+1, d); 
    else
        if  d~= size(W_ini,2)
           error('Initial W: dim of ''W_ini'' doesnot match X');
        end
        W = W_ini;  
    end
    % Initialize some variables
    eta = 0.5;                      % learning rate
    % if W is initialized randomly, we suggest that take a larger value for
    % eta in the first several iters and then reduce it, 
    % for example, eta = 5 for first 20 iters, and then eta =0.5;
    % note that eta may be changed under different batchsizes.
    max_iter = 300;                % maximum number of iterations
    momentum = 0.5;                 % initial momentum
    final_momentum = 0.75;           % final momentum
    mom_switch_iter = 150;          % iteration where momentum changes
    lamda = 1;                      %parameter for tradeoff KL1 and KL2
    mu = 0.01;                    % regularization parameter
    
    W_incs = zeros(dim+1, d);
    % Compute Z
    Y=Xone*W;
    Z=sigmf(Y,[1 0]);
    % replace 0 with eps, which is for avoiding NaN
    P = max(P, eps); 
                            
    % batch setting up
    batchsize=100;
    numbatches = floor(n / batchsize);
    
    % Iterating loop
    for iter=1:max_iter
            
            sum_Z = sum(Z .^ 2, 2); 
            Q = exp(-bsxfun(@plus, sum_Z, bsxfun(@plus, sum_Z', -2 * Z * Z')) );    % Gaussian probabilities
            Q(1:n+1:end) = 0;
            Q = Q ./ repmat(sum(Q, 2), [1 n]);
            Q = max(Q, eps);
            costs1 = sum(P .* log((P + eps) ./ (Q + eps)), 2) ./ n;              % division by n corrects for # of datapoints
            costs2 = sum(Q .* log((Q + eps) ./ (P + eps)), 2) ./ n;
            cost = lamda*sum(costs1)+(1-lamda)*sum(costs2);
            disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);

            kk = randperm(n);
        for i=1:numbatches
            disp(['numbatches ' num2str(i) '/' num2str(numbatches)]);
            batch_X = Xone( kk((i - 1) * batchsize + 1 : i * batchsize),:);
            batch_P = P(kk((i - 1) * batchsize + 1 : i * batchsize),  kk((i - 1) * batchsize + 1 : i * batchsize));
            
            % Compute Gaussian kernel for selected embedded data representation batch_X
            sum_Z = sum(Z .^ 2, 2); 
            Q = exp(-bsxfun(@plus, sum_Z, bsxfun(@plus, sum_Z', -2 * Z * Z')) );    % Gaussian probabilities
            Q(1:n+1:end) = 0;
            Q = Q ./ repmat(sum(Q, 2), [1 n]);
            Q = max(Q, eps);
            batch_Q = Q(kk((i - 1) * batchsize + 1 : i * batchsize),  kk((i - 1) * batchsize + 1 : i * batchsize));
            batch_Z = Z( kk((i - 1) * batchsize + 1 : i * batchsize),:);
            
            % Compute gradient            
            batch_ZIJ = batch_Z.*(1-batch_Z);
            %--------------KL1 ------------------
            batch_PQ = batch_P - batch_Q + batch_P' - batch_Q';
            batch_PQsum = sum(batch_PQ,2);
            dW1 = 2*batch_X'*(repmat(batch_PQsum,1,d).*batch_Z.*batch_ZIJ-batch_PQ*batch_Z.*batch_ZIJ);
            %------------end KL1-----------------
            
            %---------------KL2------------------
            if lamda ~= 1
                batch_QQP = batch_Q.*log(batch_Q./batch_P);
                batch_QQPsum = sum(batch_QQP,2);
                A = batch_Q .* repmat(batch_QQPsum,1,batchsize);
                D = A + A' - batch_QQP - batch_QQP';
 
                Dsum = sum(D,1); 
                dW2 = 2*batch_X'*(repmat(Dsum',1,d).*batch_Z.*batch_ZIJ-D'*batch_Z.*batch_ZIJ);
            %---------end KL2--------------------
            % The final gradient of KL
                dW = lamda*dW1 + (1-lamda)*dW2 + mu*W;
            else
                dW = dW1 + mu*W;
            end
            
            % Update the solution===================
            W_incs = momentum * W_incs - eta * dW;
            W = W + W_incs;
            
            if iter == mom_switch_iter
               momentum = final_momentum;
            end
            
          % Compute Z
          Y=Xone*W;
          Z=sigmf(Y,[1 0]);    
        end

    end
end
