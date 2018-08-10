function [A,B,flag,ridx] = nUpdateBudgetFIFOC(at,id,A,B,N,K)
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
    ac = A(1);
    cid = B(1);
    A(1)=[];
    B(1)=[];
    A = [A at];
    B = [B id];
    flag = 1;
    ridx = 1;
    [mi,j]=max(K(cid,B));
    A(j) = A(j)+ac;
end

end




