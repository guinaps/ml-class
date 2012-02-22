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

min_error = 1;
curr_C = 0.01;

while curr_C < 30
    curr_sigma = 0.01;

    while curr_sigma < 30
        model = svmTrain(X,
                         y,
                         curr_C,
                         @(x1, x2) gaussianKernel(x1, x2, curr_sigma));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions != yval));

        if err < min_error
            min_error = err;
            C = curr_C;
            sigma = curr_sigma;
        end

        curr_sigma *= sqrt(10);
    end

    curr_C *= sqrt(10);
end






% =========================================================================

end
