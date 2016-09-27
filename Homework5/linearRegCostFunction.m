function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%start out with the linear hypothesis (htheta from the notes)
h = X * theta;

%compute the cost function before regularization (including bias)
J = 1 / (2 * m) * sum((h - y) .^ 2);

%create a new variable to store theta without the bias
newTheta = [0 ; theta(2:size(theta), :)];

%compute the regularization of the cost function from the new theta variable
Reg = lambda * sum(newTheta .^ 2) / (2 * m);

%compute the total cost function and regularization
J = J + Reg;

%compute the gradient, the partial derivative of the regularized linear 
% regression's cost.  Do not do the summation. 
grad = ((1/m)*(X' * (h-y))) + ((lambda/m)*newTheta);

% =========================================================================

grad = grad(:);

end
