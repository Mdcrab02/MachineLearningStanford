function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


positive = find(y==1);
negative = find(y==0);
% P(y = 1|x;theta) and P(y = 0|x;theta)

plot(X(positive, :), 'k+');
plot(X(negative, :), 'ko');
%plot both using the same symbols as before


% =========================================================================



hold off;

end