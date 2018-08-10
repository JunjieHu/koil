function [An Ap at] = nUpdateKernel4(id,y,K,C,eta,An,Bn,Ap,Bp,num)
% KOIL: Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% nUpdateKernel: update alpha=[Ap An], bounded alpha, for finite buffer 
%--------------------------------------------------------------------------
% INPUT:
%       id:    id for training data x, number;
%        y:    the label for x;
%        K:    the kernel matrix
%        C:    the penalty parameter for the AUC objective function;
%      eta:    the learning rate;
%       An:    the negative alpha;
%       Bn:    the negative budget storing IDs of SV.
%       Ap:    the possitive alpha;
%       Bp:    the possitive budget storing IDs of SV.
%--------------------------------------------------------------------------

cnt = 0;
at  = 0;
f_B = [];
loss = [];
loss_idx=[];
lidx = [];
%% predict the label for x
alpha = [Ap An];
SV = [Bp Bn];
if (isempty(alpha)),
    f_xt=0;
else
    k_t = K(id,SV(:))';
    f_xt=alpha*k_t;
end

%% make the copy for An
An1 = (1-eta)*An;
%An1 = An;

lcnt = 0;
%% update the pairwise loss objective function
for i=1:size(Bn,2)
    ID=Bn(i);
    %% predict the label for the ith support in Bn
    if (isempty(alpha)),
        f_B(i)=0;
    else
        k_t = K(ID,SV(:))';
        f_B(i)=alpha*k_t;
    end
    
    ploss = 1-y*(f_xt-f_B(i));
    if(ploss>0)
        cnt = cnt+1;
        loss(1,cnt)=2*ploss;
        loss_idx(cnt)=i;
    end
end

Ap1 = (1-eta)*Ap;
% Ap1=Ap;
% % Insure that the positive sample has the weight when the negative buffer is empty
if size(Bn,2)<1  
    at = 1*eta*C*y;
    %return;
else  
    if size(Bp,2)<1 && cnt==0
        at = 1*eta*C*y;
    else
        if cnt<num
            at = eta*C*cnt*y;
            An1(loss_idx) = An1(loss_idx)-y*eta*C;
        else
            at = eta*C*num*y;
            loss = K(id,Bn(loss_idx));
            [sloss, lidx] = sort(loss,'descend');
            li = lidx(1:num);
            lossidx = loss_idx(li);
            An1(lossidx) = An1(loss_idx(lidx(1:num))) -eta*C*y ;
        end
    end
end


Ap = Ap1; 
An = An1;
end
