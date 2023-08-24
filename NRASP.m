%--------------------------------------------------------------------------
% Code for NRASP (Unsupervised Feature Selection via Nonlinear Representation and Adaptive Structure Preservation)
%--------------------------------------------------------------------------
function [index,output] = NRASP(X,...
                                alpha, ...
                                beta, ...
                                gamma, ...
                                input_layer_size, ...
                                hidden_layer1_size, ...
                                hidden_layer2_size, ...
                                MaxIter1, ...
                                MaxIter2, ...
                                activate_function, ...
                                intialize_way)

%% =================== Part 0: Initialize NNParms ===================
% 设置部分有用变量
m = size(X, 1);
n = size(X, 2);

activate_funtion = activate_function; %'sigmoid'/'tanh'/'relu' 3*1cell

% intialize_way 3*1 cell
% 初始化参数
initial_Theta0 = initializeWeights(hidden_layer1_size, input_layer_size, intialize_way{1});
initial_Theta1 = initializeWeights(hidden_layer2_size, (hidden_layer1_size + 1), intialize_way{2});
initial_Theta2 = initializeWeights(n, (hidden_layer2_size + 1), intialize_way{3});

initial_nn_params = [initial_Theta0(:) ; initial_Theta1(:) ; initial_Theta2(:)];
nn_params = initial_nn_params;

%% =================== Part 1: Start loop ===================
for iter = 1:MaxIter2
    iter

    %% =================== Part 2: Update W ===================
    costFunction =  @(p) costFun(X,...
                            alpha, ...
                            beta, ...
                            gamma, ... 
                            p, ...
                            input_layer_size, ...
                            hidden_layer1_size, ...
                            hidden_layer2_size, ...
                            activate_funtion{1}, ...
                            activate_funtion{2}, ...
                            activate_function{3});

    % try to use minFunc
    options = [];
    options.Method='lbfgs';
    options.Display='iter';

    if ~isempty(MaxIter1)
        options.MaxIter = MaxIter1;
    end

    [nn_params, ~, ~, output] = minFunc(costFunction, nn_params, options);


    % Obtain Theta1 and Theta2 back from nn_params
    first_end = hidden_layer1_size * (input_layer_size);


    Theta0 = reshape(nn_params(1:first_end), ...
                     hidden_layer1_size, (input_layer_size));


    W = Theta0';
end 

%% =================== Part 3: Return Idx ===================

SumW = sqrt(sum(W.^2, 2));
[~, index] = sort(SumW, 'descend');


