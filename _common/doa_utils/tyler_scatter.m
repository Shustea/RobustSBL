function R = tyler_scatter(X, varargin)
% Tyler's M-estimator of scatter (scale-free). Iterative fixed-point.
% X: MxK data (zero-mean assumed). Returns MxM positive-definite matrix.
    p = inputParser;
    addOptional(p,'tol',1e-6);
    addOptional(p,'maxit',200);
    addOptional(p,'eps_q',1e-12);
    addOptional(p,'ridge',1e-12);
    parse(p,varargin{:});
    tol   = p.Results.tol;
    maxit = p.Results.maxit;
    eps_q = p.Results.eps_q;
    ridge = p.Results.ridge;

    [M,K] = size(X);
    R = (X*X')/K;
    R = (R + R')/2;
    R = R / trace(R) * M;

    for it = 1:maxit
        Q = real(sum(conj(X) .* (R\X), 1));
        w = M ./ max(Q, eps_q);

        Rn = zeros(M, 'like', X);
        for k = 1:K
            Rn = Rn + w(k) * (X(:,k)*X(:,k)');
        end
        Rn = (Rn + Rn')/2;
        Rn = Rn / trace(Rn) * M;
        Rn = Rn + ridge*eye(M, 'like', Rn);

        if norm(Rn - R, 'fro')/max(norm(R, 'fro'), 1e-16) < tol
            R = Rn; return; 
        end
        R = Rn;
    end
end

