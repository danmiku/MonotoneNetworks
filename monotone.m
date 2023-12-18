global d; 
d = 10;          % dimension 
global n;
n = 100;         % sample size

global first_layer_w;
first_layer_w = zeros(n*d,d);
global first_layer_biases;
first_layer_biases = zeros(n*d,1);
global second_layer_w;
second_layer_w = zeros(n,n*d);
global second_layer_biases;
second_layer_biases = zeros(n,1);
global third_layer_w;
third_layer_w = zeros(n,n);
global third_layer_biases;
third_layer_biases = zeros(n,1);
global fourth_layer_w;
fourth_layer_w = zeros(n,1);

global data;      %data set
global labels;    %labels for data set, should be monotone


% Test interpolating network on randomly distributed points and labels
% taken from the exponential norm function
data = construct(n,d);
labels = construct_exp_labels(data,n);
construct_network();
test()



% Estimate the extrapolation error with increasing grid sizes
d = 4;

n = 2;
extrapolate()
n = 3;
extrapolate()
n = 4;
extrapolate()


% Visualize the network in d = 2
d = 2;
[x,y,z] = surface(100);
s = surf(x,y,z)
s.EdgeColor = 'none';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sdata, slabels] = preprocess(data, labels)
    [slabels,I]=sort(labels);
    sdata=data(I,:);
end

function output = apply(x)
    % APPLY applies the constructed network on a given input. Should only
    % be called after construct_network.
    %  
    % output = apply(x) computes the output of the constructed interpolaing
    % network on the input x.

    global first_layer_w;
    global first_layer_biases;
    global second_layer_w;
    global second_layer_biases;
    global third_layer_w;
    global third_layer_biases;
    global fourth_layer_w;
    
    x = x';
    x1 = first_layer_w*x-first_layer_biases;
    b1 = thresh(x1);
    
    x2 = second_layer_w*b1 - second_layer_biases;
    b2 = thresh(x2);
    
    x3 = third_layer_w*b2 - third_layer_biases;
    b3 = thresh(x3);
    
    output = fourth_layer_w*b3;
end


function construct_network()
    % CONSTRUCT_NETWORK initalizes the weights of the layers 4 networks,
    % requires a preset data set (data) and labels (labels).

    global data;
    global labels;
    [data, labels] = preprocess(data, labels);
    construct_first_layer_weights();
    construct_second_layer_weights();
    construct_third_layer_weights();
    construct_fourth_layer_weights();
end

function construct_first_layer_weights()
    global n;
    global d;
    global data;
    global first_layer_w;
    global first_layer_biases;
    
    I = eye(d);
    first_layer_w = repmat(I, n,1);
    first_layer_biases = reshape(data', n*d,1);
end

function construct_second_layer_weights()
    global n;
    global d;
    global second_layer_w;
    global second_layer_biases;
    
    I = eye(n);
    second_layer_w = kron(I, ones(1,d));
    second_layer_biases = d * ones(n, 1);
end

function construct_third_layer_weights()
    global n;
    global d;
    global third_layer_w;
    global third_layer_biases;
    
    third_layer_w = triu(ones(n));
    third_layer_biases = ones(n, 1);
end

function construct_fourth_layer_weights()
    global labels;
    global fourth_layer_w;
    
    diffs = diff(labels);
    fourth_layer_w = cat(2,labels(1),diffs);
end




function error = extrapolate()
    global d;
    global n;
    global data;
    global labels;
    
    data = make_2d_grid(n);
    labels = construct_exp_labels(data,length(data));
    n = length(data);
    
    construct_network();
    error = 0;
    for i = 1:2^d
        point = rand(1,d);
        xerror = abs(apply(point) - exp(norm(point)));
        error = max(error,xerror);
    end
    point = 0.9999999*ones(1,d);
    error = max(error,abs(apply(point) - exp(norm(point))));
end

function [x,y,z] = surface(space)
    % SURFACE Creates surface coordinates and values for plotting an
    % interpolating network.
    %  
    % [x,y,z] = surface(space) creates a space x space grid  in the (x,y)
    % planes and returns computes the z values on this grid.
    global data;
    global labels;
    global d;
    global n;
    
    d = 2;
    x = linspace(0,1,space);
    y = linspace(0,1,space);
    data = make_grid(floor(space/10));
    n = length(data);
    labels = construct_exp_labels(data, length(data));
    construct_network()
    z = zeros(length(x),length(y));
    for i = 1:length(x)
        for j = 1:length(y)
            point = [x(i),y(j)];
            z(i,j) = apply(point);
        end
    end
end


function [x,y,z] = norm_surface(space)
    % SURFACE Creates surface coordinates and values for plotting the norm
    % function
    %  
    % [x,y,z] = surface(space) creates a space x space grid  in the (x,y)
    % planes and returns computes the z values on this grid.
    
    x = linspace(0,1,space);
    y = linspace(0,1,space);
    z = zeros(length(x),length(y));
    for i = 1:length(x)
        for j = 1:length(y)
            point = [x(i),y(j)];
            z(i,j) = norm(point);
        end
    end
end

function [x,y,z] = exp_norm_surface(space)
    % SURFACE Creates surface coordinates and values for plotting the
    % exponential norm function
    %  
    % [x,y,z] = surface(space) creates a space x space grid  in the (x,y)
    % planes and returns computes the z values on this grid.
    
    x = linspace(0,1,space);
    y = linspace(0,1,space);
    z = zeros(length(x),length(y));
    for i = 1:length(x)
        for j = 1:length(y)
            point = [x(i),y(j)];
            z(i,j) = exp(norm(point));
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%functions to create multi dimensional grids of varying sizes. Not the most
%Modular option.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function grid = make_2d_grid(space)
    x = linspace(0,1,space);
    [a,b] = ndgrid(x);
    grid = [a(:), b(:)];
end

function grid = make_3d_grid(space)
    x = linspace(0,1,space);
    [a,b,c] = ndgrid(x);
    grid = [a(:), b(:), c(:)];
end

function grid = make_4d_grid(space)
    x = linspace(0,1,space);
    [a,b,c,d] = ndgrid(x);
    grid = [a(:), b(:), c(:), d(:)];
end

function grid = make_5d_grid(space)
    x = linspace(0,1,space);
    [a1,a2,a3,a4,a5] = ndgrid(x);
    grid = [a1(:),a2(:),a3(:),a4(:),a5(:)];
end

function grid = make_6d_grid(space)
    x = linspace(0,1,space);
    [a1,a2,a3,a4,a5,a6] = ndgrid(x);
    grid = [a1(:),a2(:),a3(:),a4(:),a5(:),a6(:)];
end




function grid = make_grid(space)
    % MAKE_GRID Creates a list of points evenly spaced on a grid over the
    % 2-dimensional unit square
    %  
    % grid = make_grid(space) creates a space^2 x 2 matrix of evenly spaces
    % points.
    
    x = linspace(0,1,space);
    grid = zeros(length(x)^2,2);
    for i = 1:length(x)
        for j = 1:length(x)
            grid(i + (j - 1)*length(x), 1) = x(i);
            grid(i + (j - 1)*length(x), 2) = x(j);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = thresh(x)
    % THRESH The threshold, Heaviside, or step function
    
    out = x;
    out(out >= 0) = 1;
    out(out < 0) = 0;
    %out = 0;
    %if x >= 0
    %    out = 1;
    %end
end

function error = test()
    % TEST computes the interpolation error of the constructed network
    %  
    % error = test() computes the interpolation error of the
    % constructed network when intetpolating the points in
    % the data with the given labels. Should return 0.
    
    global n;
    global data;
    global labels;
    error = 0;

    for i = 1: n
        mylabel = apply(data(i,:));
        error = error + (labels(i) - mylabel)^2;
    end 
end



function data = construct(n,d)
    % CONSTRUCT constructs a random data set in the unit cube
    %  
    % data = construct(n,d) a data uniformly distributed random data set of
    % size n over the d-dimensional unit hypercube
    data = rand(n,d);
end

function labels = construct_labels(data, n)
    % CONSTRUCT_LABELS computes labels of a monotone function for a data
    % set
    %  
    % labels = construct_labels(data, n) computes the (monotone) norm
    % function for the data set of size n
    
    labels = zeros(1,n);
    for i = 1:n
       labels(i) = norm(data(i,:));
    end
end

function labels = construct_exp_labels(data, n)
    % CONSTRUCT_EXP_LABELS computes exponential labels of a monotone function                        
    % for a data set
    %  
    % labels = construct_labels(data, n) computes the (monotone)
    % exponential of the norm function for the data set of size n
    
    labels = zeros(1,n);
    for i = 1:n
       labels(i) = exp(norm(data(i,:)));
    end
end