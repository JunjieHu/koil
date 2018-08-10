function [A,B,flag,ridx] = nUpdateBudgetRS(at,id,A,B,N,t)
% KOIL: Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)

ridx = -1;
flag = 0;
if at==0
    return
end
%Info.type = type;
n = size(B,2);
if n<N
    A = [A at];
    B = [B id];
    flag = 1;
else
    
    prob= N/t;
    z_t = random('Binomial',1,prob);
    if z_t==1,
        permt=randperm(N);
        ix=[1:permt(1)-1,permt(1)+1:N];
        A = A(ix);
        B = B(ix);
        A = [A at];
        B = [B id];
        flag = 1;
        ridx = permt(1);
    end
end

end



