function P = mvdr_doa_estimation(A, R, eps_scale)
% MVDR_DOA_ESTIMATION  MVDR spatial spectrum over a steering grid.
%   P = mvdr_doa_estimation(A, R) returns a Gx1 spectrum where A is MxG
%   (each column a steering vector) and R is MxM covariance.
%   Optional: eps_scale (default 1e-6) controls diagonal loading.
%
%   Inputs:
%     A (M x G) : steering matrix over grid (columns = steering vectors)
%     R (M x M) : sample/robust covariance
%   Output:
%     P (G x 1) : MVDR spectrum (linear scale)

    if nargin < 3 || isempty(eps_scale), eps_scale = 1e-6; end
    M   = size(R,1);
    dl  = eps_scale * trace(R) / max(M,1);
    Rl  = R + dl*eye(M);

    % Solve Ri*A stably (no explicit inverse). Prefer Cholesky if SPD.
    % Symmetrize to damp tiny asymmetries from numerics.
    [U,p] = chol((Rl+Rl')/2, 'upper');
    if p == 0
        RiA = U \ (U' \ A);     % (R^(-1))*A
    else
        RiA = Rl \ A;           % fallback if not SPD
    end

    % MVDR denominator diag(A^H * R^{-1} * A) vectorized:
    den = real(sum(conj(A) .* RiA, 1));   % 1xG
    den = max(den, 1e-12);                % floor to avoid division blow-ups
    P   = (1 ./ den).';                   % Gx1
end
