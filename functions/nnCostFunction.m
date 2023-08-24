function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer1_size, ...
                                   hidden_layer2_size, ...
                                   X, ...
                                   activate_funtion1, ...
                                   activate_funtion2, ...
                                   activate_funtion3)

% Implements the neural network cost function for a three layer
%%      根据activate参数选择函数
if ~exist('activate_funtion1', 'var') || (strcmp(activate_funtion1,'sigmoid'))
    fun1 = @(x) sigmoid(x);
    fun1_grad = @(x) sigmoidGradient(x);
elseif (strcmp(activate_funtion1,'tanh'))
    fun1 = @(x) tanh(x);
    fun1_grad = @(x) tanhGradient(x);
elseif (strcmp(activate_funtion1,'relu'))
    fun1 = @(x) relu(x);
    fun1_grad = @(x) reluGradient(x);
elseif (strcmp(activate_funtion1,'identical'))
    fun1 = @(x) identical(x);
    fun1_grad = @(x) identicalGradient(x);
end    

if ~exist('activate_funtion2', 'var') || (strcmp(activate_funtion2,'sigmoid'))
    fun2 = @(x) sigmoid(x);
    fun2_grad = @(x) sigmoidGradient(x);
elseif (strcmp(activate_funtion2,'tanh'))
    fun2 = @(x) tanh(x);
    fun2_grad = @(x) tanhGradient(x);
elseif (strcmp(activate_funtion2,'relu'))
    fun2 = @(x) relu(x);
    fun2_grad = @(x) reluGradient(x);
elseif (strcmp(activate_funtion2,'identical'))
    fun2 = @(x) identical(x);
    fun2_grad = @(x) identicalGradient(x);
end

if ~exist('activate_funtion3', 'var') || (strcmp(activate_funtion3,'sigmoid'))
    fun3 = @(x) sigmoid(x);
    fun3_grad = @(x) sigmoidGradient(x);
elseif (strcmp(activate_funtion3,'tanh'))
    fun3 = @(x) tanh(x);
    fun3_grad = @(x) tanhGradient(x);
elseif (strcmp(activate_funtion3,'relu'))
    fun3 = @(x) relu(x);
    fun3_grad = @(x) reluGradient(x);
elseif (strcmp(activate_funtion3,'identical'))
    fun3 = @(x) identical(x);
    fun3_grad = @(x) identicalGradient(x);
end   




%%      重建Theta0 Theta1 Theta2

first_end = hidden_layer1_size * (input_layer_size);
second_end = first_end +  hidden_layer2_size * (hidden_layer1_size + 1);
% Setup some useful variables
m = size(X, 1);
n = size(X, 2);

Theta0 = reshape(nn_params(1:first_end), ...
                 hidden_layer1_size, (input_layer_size));

Theta1 = reshape(nn_params((first_end+1):second_end), ...
                 hidden_layer2_size, (hidden_layer1_size + 1));

Theta2 = reshape(nn_params((second_end+1):end), ...
                 n, (hidden_layer2_size + 1));



%% === Part1   Feedforward-propogation ===


%feedforward
a0 = X;        
a0 = [a0];              %5000*400

z1 = a0*Theta0';        %5000*25

a1 = fun1(z1);
a1 = [ones(m,1) a1];    %5000*26

z2 = a1*Theta1';
a2 = fun2(z2);   
a2 = [ones(m,1) a2];    %5000*11

z3 = a2*Theta2';    
a3 = fun3(z3);          %5000*400

% Compute J

%   -----||X - f(g(X))||_F^2-----
cost = sum(sum((a3 - X).^2));
J = cost; 

%% === Part 2  implement Backpropogation & regularize ===
delta3 = (a3 - X) .* fun3_grad(z3);  %5000*400
delta2 = (delta3 * Theta2(:,2:end)) .* fun2_grad(z2); %5000*10
delta1 = (delta2 * Theta1(:,2:end)) .* fun1_grad(z1); %5000*25

Delta2 = delta3'*a2;  %400*11   
Delta1 = delta2'*a1;  %10*26    
Delta0 = delta1'*a0;  %25*400   


D0 = Delta0.*2;
D1 = Delta1.*2;
D2 = Delta2.*2;

Theta0_grad = D0; Theta1_grad = D1;   Theta2_grad = D2;


% =========================================================================

% Unroll gradients
% Grad:[Theta0 b1 Theta1 b2 Theta2 ]'
grad = [Theta0_grad(:) ; Theta1_grad(:) ; Theta2_grad(:)];


end
