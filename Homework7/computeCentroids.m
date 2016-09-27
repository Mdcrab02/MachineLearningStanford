function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

count = zeros(K, 1);

for i=1:m

  %as we loop through, increment count by one 
	count(idx(i)) += 1;
  
  %the centroid in that position (from count) has added to it the associated
  %value from centroids, which is the centroid closes to that value
	centroids(idx(i),:) += X(i,:);	
  
end;

%keep in mind K is the size from centroids (given)
for i=1:K

  %take the value of that centroid from the matrix and divide it by the count
  %value
	centroids(i,:) = centroids(i,:) / count(i);
  
end;


% =============================================================


end

