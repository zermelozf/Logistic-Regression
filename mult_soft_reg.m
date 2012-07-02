%%%%%%%% MULTIVARIATE LOGISTIC REGRESSION %%%%%%%%
%
% Training a multivariate logistic regression with 
% Newton-Raphson or simple Gradient Ascent.
%

training_mode = 2;

%%%% BUILD DATASET %%%%
nb_samples = 1000; input_dim = 2; output_dim = 10; nb_iter = 30;

x = -5 + 10*rand(nb_samples, input_dim);
x(:,end) = 1;

T = 2*(-1 + 2*rand(input_dim, output_dim));
t = softmax((x*T)')';
y = mnrnd(1,t);

%Initialization
T = 2*(-1 + 2*rand(input_dim, output_dim));

if training_mode == 1
    %%%% TRAIN WITH NEWTON-RAPHSON %%%%
    pe = zeros(nb_samples, output_dim);
    hessian = zeros(input_dim*output_dim, input_dim*output_dim);
    for iter = 1:nb_iter
        %Gradient
        grad = x'*(softmax((x*T)')' - y);
        grad = reshape(grad, input_dim*output_dim, 1);

        %Hessian
        for class = 1:output_dim
            prob = softmax((x*T)')';
            w_diag = prob(:, class).*(1-prob(:, class));
            hessian((class-1)*input_dim+1:class*input_dim, (class-1)*input_dim+1:class*input_dim) = x'*diag(w_diag)*x;
            for other_class=class+1:output_dim
                w_off_diag = -prob(:, class).*prob(:,other_class);
                hessian_off_diag = x'*diag(w_off_diag)*x;
                hessian((class-1)*input_dim+1:class*input_dim, (other_class-1)*input_dim+1:other_class*input_dim) = hessian_off_diag;
                hessian((other_class-1)*input_dim+1:other_class*input_dim, (class-1)*input_dim+1:class*input_dim) = hessian_off_diag;
            end
        end

        %Update
        if rcond(hessian) < eps,
            for i = -16:16,
                h2 = hessian.*(( 1+ 10^i)*eye(size(hessian))  + (1-eye(size(hessian))));
                if rcond(h2) > eps, break, end
            end
        hessian = h2;
        end

        delta = reshape(hessian\grad, input_dim, output_dim);
        T = T - min(0.1*iter, 1)*delta
    end

else
    %%%% TRAIN WITH GRADIENT ASCENT %%%%
    for iter=1:1000
        T = T - 0.0001*(x'*(softmax((x*T)')' - y));
    end
end

%%%% PLOTTING %%%%

tt = softmax((x*T)')';
clf;
hold on;
cmap= hsv(output_dim);
for i = 1:output_dim
    plot(x(:,1), t(:, i), 'kx')
    plot(x(:, 1), tt(:, i), 'x', 'color', cmap(i,:))
end

