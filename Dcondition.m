function dc = Dcondition(Y,D,Omega)

% compute the condition number of trained dictionary
%
% Wei Dai, Tao Xu, Wenwu Wang
% Imperial College London, University of Surrey
% wei.dai1@imperial.ac.uk  t.xu@surrey.ac.uk  w.wang@surrey.ac.uk
% October 2011

[m,n] = size(Y);
dc = zeros(3,n);
for cn = 1:n
    Dt = D(:,Omega(:,cn));
    Lambda = svd(Dt);
    dc(1,cn) = Lambda(1)/Lambda(end);
    dc(2,cn) = Lambda(1);
    dc(3,cn) = Lambda(end);
    0;
end