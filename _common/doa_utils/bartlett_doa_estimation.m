function P = bartlett_doa_estimation(A, R)
    % BARTLETT_ESTIMATION Computes the Bartlett power spectrum estimation.
    %   P = BARTLETT_ESTIMATION(A, R) returns the power spectrum P for the
    %   input data matrix A (GxM) and the covariance matrix R (MxM).
    %   The output P is a Gx1 vector where each element corresponds to the
    %   estimated power for each sensor.
    %
    %   Inputs:
    %       A - Data matrix of size GxM
    %       R - Covariance matrix of size MxM
    %
    %   Outputs:
    %       P - Estimated power spectrum of size Gx1

    AR = (R * A);         % GxM  (same as (A*R) if A were MxG)
    P  = real(sum(conj(A) .* AR, 1));   % Gx1
    P  = max(P, 1e-16);
end