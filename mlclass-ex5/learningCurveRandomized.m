function [error_train, error_val] = ...
    learningCurveRandomized(X, y, Xval, yval, lambda)
%LEARNINGCURVERANDOMIZED Generates the train and cross validation set errors
%needed to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVERANDOMIZED(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)). For each i, there
%       is a fixed number of iterations where the i examples are chosen
%       randomically and the corresponding error calculated. The resulting
%       error is the average of the errors of all iterations.
%

% Number of training examples
m = size(X, 1);

% Number of cross validation examples
mval = size(Xval, 1);

% Number of train iterations for each number of examples
iters = 50;

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for num_ex = 1:m
    error_train_iter = zeros(iters, 1);
    error_val_iter   = zeros(iters, 1);

    for i = 1:iters
        train_inds = randperm(m)(1:num_ex);
        X_subset = X(train_inds,:);
        y_subset = y(train_inds);

        val_inds = randperm(mval)(1:min(mval, num_ex));
        Xval_subset = Xval(val_inds,:);
        yval_subset = yval(val_inds);

        theta = trainLinearReg(X_subset, y_subset, lambda);

        error_train_iter(i) = linearRegCostFunction(X_subset,
                                                    y_subset,
                                                    theta,
                                                    0);
        error_val_iter(i) = linearRegCostFunction(Xval_subset,
                                                  yval_subset,
                                                  theta,
                                                  0);
    end

    error_train(num_ex) = mean(error_train_iter);
    error_val(num_ex) = mean(error_val_iter);
end

end
