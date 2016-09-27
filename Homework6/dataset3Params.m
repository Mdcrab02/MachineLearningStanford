function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%vector of possible C and sigma values
vals = [0.01 0.03 0.1 0.3 1 3 10 30]';

% matrix of congruent size with vals to store all of the different results
%   from the model in the loop below.  Here we are going to store the errors to
%   better asses the C and sigma combination
predictions = zeros(size(vals));

%the loop to go over all possible combinations of C and sigma (64 in this case)
for i=1:size(vals,1), %Outside for C

	for j=1:size(vals,1), %inside for sigma
		
    %each instance of model uses the chosen C and sigma values, calls svmTrain
    %from this application, and gaussianKernel (the function we just made in
    %gaussianKernel.m
		model = svmTrain(X, y, vals(i), @(x1, x2) gaussianKernel(x1, x2, vals(j))); 
    
    %use the model we just made and one of our given x values (after preprocessing)
    %to make a prediction
		predict = svmPredict(model,Xval);
    
    %assign into the matrix @ position i,j the prediction error
		predictions(i,j) = mean(double(predict ~= yval));
		
	end;
  
end;

%grab the values i and j from our predictions
[col, row] = min(predictions);

%from that column, find the minimum value and it's index position
[minerror, index] = min(col);


%sigma is just the index (the column value)
sigma = vals(index);
%because of how we stored these values in the matrix, the C value that minimized
% the prediction error is the value associated with the minimum value from col
% we found above
C = vals(row(index));




% =========================================================================

end
