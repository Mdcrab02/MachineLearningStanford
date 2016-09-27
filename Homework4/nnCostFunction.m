function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%initialize X
X = [ones(m,1) X];


%FORWARD PROPOGATION

%set a1 to be the term from the function h sub(theta) * x(i)
a1 = sigmoid(Theta1 * X');
%create a matrix of appropriate size and fill it with ones
a2 = [ones(m,1) a1'];

h_theta = sigmoid(Theta2 * a2');

%remember the cost function from the notes, it has another piece to the cost function puzzle
%yk(i)
%initialize that matrix of appropriate size
yk = zeros(num_labels, m);
%put into that matrix a 1 in each element
for i=1:m,
  yk(y(i),i)=1;
end

% first cost function term
first = (-yk) .* log(h_theta);

%second cost function term
second = (1-yk) .* log(1-h_theta);

%compute the cost function
%J = (1/m) * sum( sum(first - second) );

%don't do regularization on the first column because that is the bias part
%see the lecture notes on forward prop where there is a (+1) node at the top of each layer
%so create two new subset matrices that exclude the bias columns
subTheta1 = Theta1(:,2:size(Theta1,2));
subTheta2 = Theta2(:,2:size(Theta2,2));

%compute regularization
Reg = lambda  * ( sum( sum( subTheta1.^ 2 )) + sum( sum( subTheta2.^ 2 ))) / (2*m);

%compute the total cost function with regularization added
J = (1/m) * sum( sum(first - second) ) + Reg;



%BACKWARD PROPOGATION

%okay so let's go backwards this time
for t=1:m,

  %grab all of the weights from the last layer
	a1 = X(t,:);
  
  %multiply our value of theta1 from the hypothesis times each element
	z2 = Theta1 * a1';

  %now this activation layer uses our previous computation for the weights  
	a2 = sigmoid(z2);
  
  %missed this part first run, from here on back we have to add a bias layer
	a2 = [1 ; a2];

  %now we compute the weights similar to above
	z3 = Theta2 * a2;

  %this is the final activation layer going back (remember backprop)  
  %this is the same as the hypothesis of theta
	a3 = sigmoid(z3);
	
  %add back a bias layer (remember the +1 nodes in the diagram)
	z2=[1; z2];

  %errors working back from the last layer working backwards
  %so delta3 = a(4)j - yj from the notes.  we are computing the difference between
  %a3 (activation nodes from layer 3) and the appropriate element from yk
	delta_3 = a3 - yk(:,t);
  
  %again from the notes, the next layers errors are the transpose of that layer's theta
  %times the errors from the previous layer (3 in this case) multiplied by g'(z(2))
  %which here is sigmoidGradient
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2);

  %now, take only take columns 2 and forward from this matrix (remember this nn only has
  %3 layers) and we only do delta terms for layers beyond the first
	delta_2 = delta_2(2:end); 

  %in the notes, these theta grads are represented with capitaldelta
  %capitaldelta(l)ij = delta(l)ij + a(l)j * delta(l+1)j
  %remember, capdeltas are used here as accumulators
  %for some reason octave does not like a*delta from the notes... so I did delta*a'
	Theta2_grad = Theta2_grad + delta_3 * a2';
  
  %don't transpose a1, even though the notes make no specific mention as to why not
	Theta1_grad = Theta1_grad + delta_2 * a1;

end;

%training accuracy is 10.02...
%yep, for numerical values outside x in (0,1), nnets are sensitive to regularization
%notes say that D(l)ij = 1/m * capdelta(l)ij + lambda*theta(l)ij if j=0

%REGULARIZATION

  %for all rows in the first column only, element-wise division by m
  %where m is the number of training examples
	Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
  
  %for all rows in the first column only, element-wise division by m	
  %where m is the number of training examples
	Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
  
  %for all rows in the second through last columns, element-wise division
  %of m + lambda/m * the value of that particular element
	Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));
	
	Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients (into vectors)
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
