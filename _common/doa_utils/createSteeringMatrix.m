function A = createSteeringMatrix(M, d, lambda, theta)
    % createSteeringMatrix generates a steering matrix for an array.
    % M: Number of array elements.
    % d: Distance between array elements.
    % lambda: Wavelength of the signal.
    % theta: Angles of arrival (in radians) as a row vector.
    % Returns a steering matrix A of size [G x M], where G is the number of angles.
    
    theta = theta(:).';                  % [1 x G]
    m = (0:M-1).';                       % [M x 1]
    k = 2*pi/lambda;
    A = exp(1j * k * d * (sin(theta) .* m));
end
