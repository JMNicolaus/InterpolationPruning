function [P, rho] = DEIM(u)
% uses the DEIM algoirthm to calculate indices for interpolation by
% POD-DEIM
% rho will be the resulting vector of those indices;
% the columns of u are linear independet vectors

if rank(u) ~= min(size(u))
  print('Rang von U ist nicht voll!')
end
[n,m]=size(u);
rho_vec = zeros(m,1);
[~,rho_l] = max(abs(u(:,1)));
U = zeros(n,m);
U(:,1) = u(:,1);
P = zeros(n,m); P(rho_l,1)=1;%m->1
rho_vec(1) = rho_l;

for ll=2:m
  u1 = u(:,ll);
  c=(U(rho_vec(1:ll-1),1:ll-1))\(u1(rho_vec(1:ll-1)));
  r=u1-U(:,1:ll-1)*c;
  [~,rho_l]=max(abs(r));
  U(:,ll)=u1;%=[U u1];
  P(rho_l,ll)=1;
  %P=[P zeros(n,1)]; P(rho_l,end)=1;
  rho_vec(ll)= rho_l;
end
rho = rho_vec;
end






%   n = length(u(:,1));
%   rho_it = zeros(n,1);
%   [~,rho_it(1)] = max(abs(u(:,1)));
%   U(:,1) = u(:,1);
%   P(:,1) = zeros(n,1);
%   P(rho_it(1),1) = 1;
%   for l=2:length(U)
%     c = (P'*U)\(P'*u(:,l));
%     r = u(:,l)-U*c;
%     [~,rho_l]=max(abs(r));
%     U =[U u(:,l)];
%     P=[P zeros(n,1)];
%     P(rho_l,l) = 1;
%     rho_it(l)=rho_l;
%   end
%   rho=rho_it;