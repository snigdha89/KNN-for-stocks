# KNN-for-stocks

Implementation of k-NN classifier. For each week, your feature set is
(mu; sigma) for that week. Use your labels (you will have 52 labels
per year for each week) from year 1 to train your classifier and
predict labels for year 2.

1. take k = 3; 5; 7; 9; 11. For each value of k compute the
accuracy of your k-NN classifier on year 1 data. On x axis
 plot k and on y-axis you plot accuracy. What is the
optimal value of k for year 1 calculate.
2. use the optimal value of k from year 1 to predict labels for
year 2. What is your accuracy calculate
3. using the optimal value for k from year 1, compute the
confusion matrix for year 2
4. what is true positive rate (sensitivity or recall) and true
negative rate (specificity) for year 2?
5. implement a trading strategy based on your labels for year
2 and compare the performance with the "buy-and-hold"
strategy. Which strategy results in a larger amount at the
end of the year?
