function P = music_doa_estimation(A, R, D, eps_scale)
% MUSIC_DOA_ESTIMATION  MUSIC spatial spectrum over a steering grid.
%   P = music_doa_estimation(A, R, D) where:
%       A (M x G): steering matrix (cols = steering vectors for each grid angle)
%       R (M x M): covariance (Hermitian)
%       D: number of sources
%   Optional:
%       eps_scale (default 1e-6): diagonal loading scale (trace(R)/M)
%   Output:
%       P (G x 1): MUSIC pseudo-spectrum (linear)
%
%   Note: This implementation forms the noise subspace via EVD and computes
%   denom = ||E_n^H a(theta)||^2 for all grid vectors a(theta), vectorized.

    if nargin < 4 || isempty(eps_scale), eps_scale = 1e-6; end

    M = size(R,1);
    G = size(A,2);
    if size(A,1) ~= M
        error('A must be MxG (columns are steering vectors).');
    end
    if D < 0 || D >= M
        error('D must satisfy 0 <= D < M.');
    end

    % Diagonal loading (stabilize near-singular cases)
    dl = eps_scale * trace(R) / max(M,1);
    Rl = R + dl*eye(M);

    % Hermitianize for numerical stability
    Rl = (Rl + Rl')/2;

    % Eigen-decomposition (ascending -> sort descending)
    [V, S] = eig(Rl, 'vector');
    [~, idx] = sort(real(S), 'descend');

    % Noise subspace (columns): eigenvectors after the top D signal eigvecs
    En = V(:, idx(D+1:end));   % M x (M-D)

    % Compute ||E_n^H a||^2 for all grid steering vectors (vectorized)
    % En' * A -> (M-D) x G ; take column-wise squared 2-norms
    EA   = En' * A;                    % (M-D) x G
    denom = sum(abs(EA).^2, 1);        % 1 x G

    % Guard against tiny/negative numerical values
    denom = max(real(denom), 1e-16);

    % MUSIC pseudo-spectrum
    P = (1 ./ denom).';                % G x 1
end
