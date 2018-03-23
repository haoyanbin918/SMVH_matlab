function [W, Z] = SMVH_graddesc_stan(X, P, d)
%%
% This gradient descent function uses the standard gradient descent
% strategy.
%
% [W,Z] = SMVH_graddesc_stan(X, P, d)
%
% The high-dimensional datapoints are specified by X. 
% The target hash code dimensionality is specified in d (default = 100).
% The function returns the embedded points with relaxed hash codes in Z and combination
% coefficients W and bias term b in W.
%
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
    % Initialize some variables
    [n, dim] = size(X);                 % number of instances
    Xone = [X, ones(n,1)];                % add one to X
    eta = 0.05;                      % learning rate
    max_iter = 1200;                % maximum number of iterations
    momentum = 0.5;                 % initial momentum
    final_momentum = 0.75;           % final momentum
    mom_switch_iter = 250;          % iteration where momentum changes
    lamda = 0.9;                      %parameter for tradeoff KL1 and KL2
    mu = 0.01;                    % regularization parameter
    % Note: if the delta-bar-delta strategy is used, user can set mu = 0.0;
    % mu = 0.0;
    
    % Initialize combination coefficients randomly (close to origin),
    % bias b is added in the W (the last row).
    W = 0.00001*rand(dim+1, d);
    W_incs = zeros(dim+1, d);
    % Compute Z
    Y=Xone*W;
    Z=sigmf(Y,[1 0]);
    % replace 0 with eps, which is for avoiding NaN
    P = max(P, eps); 
    % ----------------------------------------------------------
    % -- The delta-bar-delta strategy can accelerate the cost --
    % -- reduction. --
    % ----------------------------------------------------------
    % gains = ones(size(W));    
    % min_gain = .01;            % minimum gain for delta-bar-delta
    % jitter = 0.3;                   % initial jitter
    % jitter_decay = 0.99;            % jitter decay
    
    % Iterating loop
    for iter=1:max_iter

        % Compute Gaussian kernel for embedded data representation Z
        sum_Z = sum(Z .^ 2, 2);                                                     % precomputation for pairwise distances
        Q = exp(-bsxfun(@plus, sum_Z, bsxfun(@plus, sum_Z', -2 * Z * Z')));    % Gaussian probabilities
        Q(1:n+1:end) = 0;
        Q = Q ./ repmat(sum(Q, 2), [1 n]);
        Q = max(Q, eps);
        
        % Compute cost function between P and Q
        if ~rem(iter, 5)
            costs1 = sum(P .* log((P + eps) ./ (Q + eps)), 2) ./ n;        
            costs2 = sum(Q .* log((Q + eps) ./ (P + eps)), 2) ./ n;
            cost = lamda*sum(costs1)+(1-lamda)*sum(costs2);
            disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
        end
        
        % Compute gradient
        ZIJ = Z .* (1-Z);
        %--------------KL1 ------------------
        PQ = P - Q + P' - Q';
        PQsum = sum(PQ,2);
        dW1 = 2*Xone'*(repmat(PQsum,1,d).*Z.*ZIJ-PQ*Z.*ZIJ);
        
        clear PQ;
        clear PQsum;
        %------------end KL1-----------------
        %---------------KL2------------------
        if lamda ~= 1
            QQP = Q.*log(Q./P);
            QQPsum = sum(QQP,2);
            A = Q.*repmat(QQPsum,1,n);
            D = A + A' - QQP - QQP';
            
            clear QQP;
            clear QQPsum;
            clear A;
            
            Dsum = sum(D,1); 
            dW2 = 2*Xone'*(repmat(Dsum',1,d).*Z.*ZIJ-D'*Z.*ZIJ);
            
            clear D;
            clear Dsum;
        %---------end KL2--------------------
        % The final gradient of KL
            dW = lamda*dW1+(1-lamda)*dW2+mu*W;
            
        else
            dW = dW1+mu*W;
        end
        
        clear ZIJ;
        clear Z;
        clear Q;
            
        % Update the solution
        % ----------------------------------------------------------
        % ----------------------------------------------------------
        % -- The delta-bar-delta strategy can accelerate the cost --
        % -- reduction. Users can replace the standard gradient --
        % -- descent method by this strategy. --
        % ----------------------------------------------------------
        % gains = (gains + .2) .* (sign(dW) ~= sign(W_incs)) ...   
        %      + (gains * .8) .* (sign(dW) == sign(W_incs));
        % gains(gains < min_gain) = min_gain;
        
        % Perform the gradient search
        % ---Use delta-bar-delta strategy to perform the gradient------
        % W_incs = momentum * W_incs - eta *(gains .* dW);
        % W = W + W_incs;
	    % W = W + jitter * randn(size(W));
        % W = bsxfun(@minus, W, mean(W, 1));
        % Reduce jitter over time and change momentum
        % jitter = jitter * jitter_decay;
        %--- delta-bar-delta strategy end------------------------------
        %--------------------------------------------------------------
        
        % Perform the gradient search by the gradient descent method
        W_incs = momentum * W_incs - eta * dW;
        W = W + W_incs;
        
        % Compute Z
        Y=Xone*W;
        Z=sigmf(Y,[1 0]);
        
        if iter == mom_switch_iter
            momentum = final_momentum;
        end
    end
end