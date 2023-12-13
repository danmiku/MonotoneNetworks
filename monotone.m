d = 2;     % dimension 
n = 600;   % sample size



data = construct(n,d);
labels = construct_labels(data,n);

%test(data,labels,n,d)

[x,y,z] = surface(n);
s = surf(x,y,z)
s.EdgeColor = 'none';

function [x,y,z] = surface(space)
    % SURFACE Creates surface coordinates and values for plotting an
    % interpolating network.
    %  
    % [x,y,z] = surface(space) creates a space x space grid  in the (x,y)
    % planes and returns computes the z values on this grid.
    
    d = 2;
    x = linspace(0,1,space);
    y = linspace(0,1,space);
    data = make_grid(floor(space/10));
    labels = construct_labels(data, length(data));
    z = zeros(length(x),length(y));
    for i = 1:length(x)
        for j = 1:length(y)
            point = [x(i),y(j)];
            z(i,j) = monotone_network(point,length(data),d,data,labels);
        end
    end
end

function grid = make_grid(space)
    % MAKE_GRID Creates a list of points evenly spaced on a grid over the
    % 2-dimensional unit square
    %  
    % grid = make_grid(space) creates a space^2 x 2 matrix of evenly spaces
    % points.
    
    x = linspace(0,1,space);
    length(x)
    grid = zeros(length(x)^2,2);
    for i = 1:length(x)
        for j = 1:length(x)
            grid(i + (j - 1)*length(x), 1) = x(i);
            grid(i + (j - 1)*length(x), 2) = x(j);
        end
    end
end

function out = thresh(x)
    % THRESH The threshold, Heaviside, or step function
      
    out = 0;
    if x >= 0
        out = 1;
    end
end

function error = test(data,labels,n,d)
    % TEST computes the interpolation error of the constructed network
    %  
    % error = test(data,labels,n,d) computes the interpolation error of the
    % constructed network when intetpolating the n d-dimensional points in
    % the data with the
    % given labels. Should return 0.
    
    error = 0;
    for i = 1: n
        mylabel = monotone_network(data(i,:),n,d,data,labels);
        error = error + (labels(i) - mylabel)^2;
    end 
end


function output = monotone_network(input,n,d,data,labels)
    % MONOTONE_NETWORK constructs and evalautes an interpolating network
    %  
    % output = monotone_network(input,n,d,data,labels) returns the output
    % of the interpolating on the given input. The network is constructed
    % by interpolating the n d-dimensional points in data with the given
    % labels
    
    [slabels,I]=sort(labels);
    sdata=data(I,:);
    
    first = first_layer_output(input, sdata, n, d);
    second = second_layer_output(first, n, d);
    third = third_layer_output(second, n);
    output = fourth_layer_output(third, slabels, n);
end
function out = first_layer_neuron(x,j,point,d)
    index = mod(j,d) + 1;
    xcoord = x(index);
    pcoord = point(index);
    out = thresh(xcoord - pcoord);
end

function  first_layer = first_layer_output(x, data, n, d)
        first_layer = zeros(1,n*d);
        for j = 1: n*d
           first_layer(j) = first_layer_neuron(x,j,data(ceil(j/d),:),d);
        end
end

function out = second_layer_neuron(x,i,d)
        out = thresh(sum(x(d*(i-1) + 1: d*i))- d);
end

function second_layer = second_layer_output(x, n, d)
        second_layer = zeros(1,n);
        for i = 1: n
           second_layer(i) = second_layer_neuron(x,i,d);
        end
end

function out = third_layer_neuron(x,i,n)
        out = thresh(sum(x(i:n))- 1);
end

function third_layer = third_layer_output(x, n)
        third_layer = zeros(1,n);
        for i = 1: n
           third_layer(i) = third_layer_neuron(x,i,n);
        end
end

function out = fourth_layer_output(x,labels,n)
        out = labels(1)*x(1);
        for i = 2:n
           out = out + x(i)*(labels(i) - labels(i-1)); 
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