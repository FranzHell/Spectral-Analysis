function y = gaussKernel(sigma, dt)
% y = gaussKernel( sigma, dt) creates a vector that 
% contains a Gaussian kernel in its center. The vector has the duration
% of 8 times the standard deviation.
%
% 'sigma' defines the standard deviation of the gaussian.
% 'dt' the temporal resolution, i.e. the stepsize or sampling interval
%
% by Jan Grewe, no warrenty!


x = -4 * sigma:dt:4 * sigma;
y = exp(-0.5 * (x ./ sigma) .^ 2) / sqrt(2 .* pi) ./ sigma;