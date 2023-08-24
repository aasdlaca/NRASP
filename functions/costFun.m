function [J,grad] = costFun(X, ...
                            alpha, ...
                            beta, ...
                            gamma, ...
                            nn_params, ...
                            input_layer_size, ...
                            hidden_layer1_size, ...
                            hidden_layer2_size, ...
                            activate_funtion1, ...
                            activate_funtion2, ...
                            activate_funtion3)
                        
% feature selection cost Funtion
% fun_name = 'sigmoid';

fun1_name = activate_funtion1;
fun2_name = activate_funtion2;
fun3_name = activate_funtion3;

%% 重建Theta0<->W

first_end = hidden_layer1_size * (input_layer_size);
W = reshape(nn_params(1:first_end), ...
                 hidden_layer1_size, (input_layer_size));
W = W';

m = size(X, 1);
n = size(X, 2);


%% ====== update S In costFun ======

% --- calculate (-beta/(2gamma))||W^T * x_i - W^T * x_j||^2_2 ---
D = EuDist2(X*W,'',0);  % 
C = ((-beta)/(2*gamma))*D;
S = exp(C) ./ sum(exp(C),2);
% LapMatrix = diag(sum(S,2))-S; 

tmpS = ((S+S')./2);
LapMatrix = diag(sum(tmpS,2))-tmpS; 

%% Compute J1 and grad1
[J1 grad1] = nnCostFunction(nn_params, input_layer_size, hidden_layer1_size, ...
                               hidden_layer2_size, X, fun1_name, fun2_name, fun3_name);


%% Compute J2 and grad2
J2 = alpha * sum(sqrt(sum(W.^2, 2)));

% grad2
temp = 2 * sqrt(sum(W.^2, 2));
temp = max(temp, eps);
Q = diag(1./temp);

grad2 = 2 * alpha * Q * W; 
grad2 = grad2';


%% Compute J3 and grad3
J3 = beta * trace(W' * X' * LapMatrix * X * W);

grad3 = 2 * beta * (X' * LapMatrix * X * W);
grad3 = grad3';

% 更新W0的梯度
% grad2和grad3只和W0有关
grad1(1:first_end) = grad1(1:first_end) + grad2(:) + grad3(:);


%% Compute J4 and grad4
J4 = gamma * sum(sum(S.*log(S)));
grad4 = 0;          % 这一项对W和b没有导数，求导为0

J = J1 + J2 + J3 + J4;
grad = grad1;