function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%matrix to store cluster assignments
c = zeros(K,1);


for i=1:size(X,1)

    for k=1:K
    
      %store into c the k value computed from the PCA function
      c(k) = sum((X(i,:) - centroids(k,:)).^2);	
      
    end;
  
  %grab the smallest k value and it's index position from c  
  [val,index] = min(c);
    
  %stick that index value (from above) into the idx matrix so it knows which
  %associated centroid has the lowest value of k for that training example
  idx(i) = index;
    
end;


% =============================================================

end

