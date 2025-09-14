function R = scm_scatter(X, varargin)
% calculates SCM (sample covariance matrix)
% X: MxK data (zero-mean assumed). Returns MxM positive-(semi)definite matrix.
    p = inputParser;
    addOptional(p,'ridge',1e-12);
    parse(p,varargin{:});
    ridge = p.Results.ridge;

    K = size(X,2);
    R = (X*X')/K;
    R = R + ridge*eye(size(R), 'like', R);
end

