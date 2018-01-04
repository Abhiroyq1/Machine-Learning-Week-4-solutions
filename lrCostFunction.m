function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
grad = zeros(size(theta));
v1=log(sigmoid(X*theta));
v2=log(1-sigmoid(X*theta));
J= -1/m*((y'*v1)+((1-y)'*v2))+((lambda/(2*m))*sum(theta(2:size(theta)).^2));

%  grad = grad + 1/m*(X'*(sigmoid(X*temp)-y)) + (lambda/m)*temp;
%% As grad(1) only takes the first value of X
%% ?J(?)/?? = 1/m(sum_iTOm(h?x(i)-y(i))*x(i)(0)

grad(1)= 1/m*(X(:,1)'*(sigmoid(X*theta)-y))

%% rest of grad takes rest all values of X
%% ?J(?)/?? = 1/m(sum_iTOm(h?x(i)-y(i))*x(i)(j)+ lambda/m * theta(j) where j>=1

grad(2:end)= 1/m*(X(:,2:end)'*(sigmoid(X*theta)-y))+(lambda/m)*theta(2:end)

% =============================================================

grad = grad(:);

end
