%this is to test problem 3, which I am certain has no right answer
function quiz2(D,x)
  v = zeros(10,1);
  for i = 1:10
    for j = 1:10
      v(i) = v(i) + D(i, j) * x(j);
    end
  end

 