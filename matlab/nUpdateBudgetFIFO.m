function [A,B,flag,ridx] = nUpdateBudgetFIFO(at,id,A,B,N)
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
    A(1)=[];
    B(1)=[];
    A = [A at];
    B = [B id];
    flag = 1;
    ridx = 1;
end

end




