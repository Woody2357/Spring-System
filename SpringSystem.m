Rhoop = 3; % the radius of the hoop
r0 = 1; % the equilibrial length of the springs
kappa = 1; % the spring constant
Nnodes = 21;
A = zeros(Nnodes,Nnodes); % spring adjacency matrix
% vertical springs
for k = 1 : 3
    A(k,k+4) = 1;
end
for k = 5 : 7  
    A(k,k+5) = 1;
end
for k = 10 : 12  
    A(k,k+5) = 1;
end
for k = 15 : 17  
    A(k,k+4) = 1;
end
% horizontal springs
for k = 4 : 7
    A(k,k+1) = 1;
end
for k = 9 : 12  
    A(k,k+1) = 1;
end
for k = 14 : 17  
    A(k,k+1) = 1;
end
% symmetrize
Asymm = A + A';
%indices of nodes on the hoop
ind_hoop = [0,3,8,13,18,19,20,17,12,7,2,1] + 1;
Nhoop = length(ind_hoop);
% indices of free nodes (not attached to the hoop)
ind_free = [4,5,6,9,10,11,14,15,16] + 1;
Nfree = length(ind_free);
% list of springs
[springs_0,springs_1] = ind2sub([Nnodes,Nnodes],find(A));
springs = [springs_0';springs_1'];

Nsprings = length(springs_0);
% maps indices of nodes to their indices in the gradient vector

%% Initialization

% Initial angles for the nodes are uniformly distributed around the range of 2*pi
% startting from theta0 and going counterclockwise
theta0 = 2*pi/3;
theta = theta0 + linspace(0,2*pi,Nhoop+1)';
theta(end) = [];
% Initial positions
pos = zeros(Nnodes,2);
pos(ind_hoop,1) = Rhoop*cos(theta);
pos(ind_hoop,2) = Rhoop*sin(theta);
pos(ind_free,1) = [-1.,0.,1.,-1.,0.,1.,-1.,0.,1.]';
pos(ind_free,2) = [1.,1.,1.,0.,0.,0.,-1.,-1.,-1.]'; 

% Initiallize the vector of parameters to be optimized
vec = [theta;pos(ind_free,1);pos(ind_free,2)]; % a column vector with 30 components

draw_spring_system(pos,springs,Rhoop,ind_hoop,ind_free);

gradient = @(vec)compute_gradient(vec,Asymm,r0,kappa,Rhoop,ind_hoop,ind_free);
func = @(vec)Energy(vec,springs,r0,kappa,Rhoop,ind_hoop,ind_free);

%% optimization

%% Method 1: Gradient Descent
max_iter = 1000;
alpha = 0.01;
tol = 1e-6;

vec_gd = vec;
energy_gd = zeros(max_iter,1);
grad_norm_gd = zeros(max_iter,1);

for iter = 1:max_iter
    E = func(vec_gd);
    grad_E = gradient(vec_gd);
    
    energy_gd(iter) = E;
    grad_norm_gd(iter) = norm(grad_E);
    
    if grad_norm_gd(iter) < tol
        break;
    end
    
    vec_gd = vec_gd - alpha * grad_E;
end

[theta_gd, pos_gd] = vec_to_pos(vec_gd, Rhoop, ind_hoop, ind_free);

figure;
subplot(2,1,1);
plot(1:iter, energy_gd(1:iter));
xlabel('Iteration');
ylabel('Energy');
title('Gradient Descent: Energy vs Iteration');

subplot(2,1,2);
semilogy(1:iter, grad_norm_gd(1:iter));
xlabel('Iteration');
ylabel('Gradient Norm');
title('Gradient Descent: Gradient Norm vs Iteration');

draw_spring_system(pos_gd, springs, Rhoop, ind_hoop, ind_free);
title('Spring System after Gradient Descent');

fprintf('Gradient Descent Results:\n');
fprintf('Final Energy: %f\n', func(vec_gd));
fprintf('Final Gradient Norm: %f\n', norm(gradient(vec_gd)));
fprintf('Positions of Nodes:\n');
disp(pos_gd);

%% Method 2: Conjugate Gradient
vec_cg = vec;
energy_cg = zeros(max_iter,1);
grad_norm_cg = zeros(max_iter,1);

grad_E = gradient(vec_cg);
d = -grad_E;

for iter = 1:max_iter
    E = func(vec_cg);
    grad_E_prev = grad_E;
    
    energy_cg(iter) = E;
    grad_norm_cg(iter) = norm(grad_E);
    
    if grad_norm_cg(iter) < tol
        break;
    end
    
    alpha = 1;
    c = 1e-4;
    rho = 0.9;
    while func(vec_cg + alpha * d) > E + c * alpha * grad_E' * d
        alpha = rho * alpha;
    end
    
    vec_cg = vec_cg + alpha * d;
    grad_E = gradient(vec_cg);
    beta = (grad_E' * grad_E) / (grad_E_prev' * grad_E_prev);
    d = -grad_E + beta * d;
end

[theta_cg, pos_cg] = vec_to_pos(vec_cg, Rhoop, ind_hoop, ind_free);

figure;
subplot(2,1,1);
plot(1:iter, energy_cg(1:iter));
xlabel('Iteration');
ylabel('Energy');
title('Conjugate Gradient: Energy vs Iteration');

subplot(2,1,2);
semilogy(1:iter, grad_norm_cg(1:iter));
xlabel('Iteration');
ylabel('Gradient Norm');
title('Conjugate Gradient: Gradient Norm vs Iteration');

draw_spring_system(pos_cg, springs, Rhoop, ind_hoop, ind_free);
title('Spring System after Conjugate Gradient');

fprintf('Conjugate Gradient Results:\n');
fprintf('Final Energy: %f\n', func(vec_cg));
fprintf('Final Gradient Norm: %f\n', norm(gradient(vec_cg)));
fprintf('Positions of Nodes:\n');
disp(pos_cg);

%%
function draw_spring_system(pos,springs,R,ind_hoop,ind_free)
% draw the hoop 
figure;
hold on;
t = linspace(0,2*pi,200);
plot(R*cos(t),R*sin(t),'linewidth',5,'color','r');
% plot springs
Nsprings = size(springs,2);
for k = 1 : Nsprings
    j0 = springs(1,k);
    j1 = springs(2,k);
    plot([pos(j0,1),pos(j1,1)],[pos(j0,2),pos(j1,2)],'linewidth',3,'color','k');
end
% plot nodes
plot(pos(ind_hoop,1),pos(ind_hoop,2),'.','Markersize',100,'Color',[0.5,0,0]);
plot(pos(ind_free,1),pos(ind_free,2),'.','Markersize',100,'Color','k');
set(gca,'Fontsize',20);
daspect([1,1,1]);
end

%% 
function grad = compute_gradient(vec,Asymm,r0,kappa,R,ind_hoop,ind_free)
    [theta,pos] = vec_to_pos(vec,R,ind_hoop,ind_free);    
    Nhoop = length(ind_hoop);
    g_hoop = zeros(Nhoop,1); % gradient with respect to the angles of the hoop nodes
    Nfree = length(ind_free);
    g_free = zeros(Nfree,2); % gradient with respect to the x- and y-components of the free nodes
    for k = 1 : Nhoop
        ind = find(Asymm(ind_hoop(k),:)); % index of the node adjacent to the kth node on the hoop
        rvec = pos(ind_hoop(k),:) - pos(ind,:); % the vector from that adjacent node to the kth node on the hoop
        rvec_length = norm(rvec); % the length of this vector
        g_hoop(k) = (rvec_length - r0)*R*kappa*(rvec(1)*(-sin(theta(k))) + rvec(2)*cos(theta(k)))/rvec_length;
    end
    for k  = 1 : Nfree
        ind = find(Asymm(ind_free(k),:)); % indices of the nodes adjacent to the kth free node
        Nneib = length(ind);
        for j = 1 : Nneib
            rvec = pos(ind_free(k),:) - pos(ind(j),:); % the vector from the jth adjacent node to the kth free node 
            rvec_length = norm(rvec);  % the length of this vector
            g_free(k,:) = g_free(k,:) + (rvec_length - r0)*R*kappa*rvec/rvec_length;
        end
    end
    % return a single 1D vector
    grad = [g_hoop;g_free(:,1);g_free(:,2)];
end

%%
function E = Energy(vec,springs,r0,kappa,R,ind_hoop,ind_free)
    [~,pos] = vec_to_pos(vec,R,ind_hoop,ind_free);
    Nsprings = size(springs,2);
    E = 0.;
    for k =1 : Nsprings
        j0 = springs(1,k);
        j1 = springs(2,k);
        rvec = pos(j0,:) - pos(j1,:);
        rvec_length = norm(rvec);       
        E = E + kappa*(rvec_length - r0)^2;
    end
    E = E*0.5;
end

%%
function [theta,pos] = vec_to_pos(vec,R,ind_hoop,ind_free)
    Nhoop = length(ind_hoop);
    Nfree = length(ind_free);
    Nnodes = Nhoop + Nfree;
    theta = vec(1:Nhoop);
    pos = zeros(Nnodes,2);
    pos(ind_hoop,1) = R*cos(theta);
    pos(ind_hoop,2) = R*sin(theta);
    % positions of the free nodes
    pos(ind_free,1) = vec(Nhoop+1:Nnodes);
    pos(ind_free,2) = vec(Nnodes+1:end); 
end