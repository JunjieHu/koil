function [Model, AUC, Accuracy,err_count, run_time, mistakes, mistakes_idx, SVs, TMs, AUC_list, Acc_list] = nKOIL_RSC( id_train, id_test, Y, K, options  )
% KOIL: Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% 
%--------------------------------------------------------------------------
% INPUT:
%        X:    training data, e.g., X(t,:) denotes for t-th instance;
%        Y:    the label vector, e.g., Y(t) is the label of t-th instance;
%   option:    contain lambda, eta, Np and Nn
%   lambda:    the regularization parameter for the AUC objective function;
%      eta:    the learning rate;
%       Np:    the maximum possitive budget size;
%       Nn:    the maximum negative budget size.
%--------------------------------------------------------------------------

%% initialize parameters
Ap=[];
An=[];
App=[];
Ann=[];
apnum=0;
annum=0;
Bp=[];
Bn=[];
n = 0;
p = 0;
bset=zeros(size(id_train,2),1);
bbest = 0;
TMs=[];
idx=[];
t_tick=options.t_tick;
err_count=0;
run_time=0;
mistakes=0;
mistakes_idx=[];
SVs=[];
TMs=[];
Ds = Inf;

alpha = [];
SV = [];
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
err_count = 0;
pid_list = [];
nid_list = [];
pid = [];
nid = [];
AUC_list=[];
Acc_list=[];
passid=[];
auc=[];
acc=[];

%% tuned prarameters
eta    = options.eta;
C  = options.C_KOIL;
Np = options.Num_p;
Nn = options.Num_n;
k1 = options.k1;
k2 = options.k2;

ratio = 0;
ratio1 = 0;

%% loop
tic
ratio=0;
for t = 1:size(id_train,1),
    id = id_train(t);
    if t==100||t==120||t==150||t==420||t==720 %t==10||t==20||t==31||
        t;
    end
           
    %% get the predicted lable for x
    alpha = [Ap An];
    SV = [Bp Bn];
    if(isempty(alpha))
        f_t = 0;
    else
        f_t = alpha*(K(id,SV(:))')-bbest;
    end
    if (f_t*Y(id)<=0)
        err_count = err_count+1;
    end
    
    y_t=Y(id);  
    if y_t==+1,
        p  = p+1;     
        %Cp = C*max(1,n/Nn);
        Cp = C;
        [An,Ap,a(t)] = nUpdateKernel4(id,y_t,K,Cp,eta,An,Bn,Ap,Bp,k1);
        [Ap,Bp,flag,ridx] = nUpdateBudgetRSC(a(t),id,Ap,Bp,Np,p,K);
        if(flag==1)
            if(ridx~=-1)
                App(ridx)=a(t);
            else 
                App = [App a(t)];
            end
        end            
    else
        n  = n+1;         
%         Cn = C*max(1,p/Np);
        Cn = C;
        [Ap,An,a(t)]  = nUpdateKernel4(id, y_t,K,Cn,eta,Ap,Bp,An,Bn,k1);        
        [An,Bn,flag,ridx] = nUpdateBudgetRSC(a(t),id,An,Bn,Nn,n,K);
        if(flag==1)
            if(ridx~=-1)
                Ann(ridx)=a(t);
            else 
                Ann = [Ann a(t)];
            end
        end
    end
    Ann = Ann*(t-1)/t+An/t;
    App = App*(t-1)/t+Ap/t;
    
    if(~isempty(Ap)&&~isempty(An))
        ratio = abs(sum(An)/sum(Ap));
        ratio1 = abs(sum(Ann)/sum(App));
    end
    rtt(t) = ratio1;
    rt(t)=ratio;      
    
    [bset(t) Acc_SV(t)] = nUpdateB(Ap,Bp,An,Bn,K);
    bbest = bset(t)
  
    %% test performance
    alpha = [Ap An];
    SV = [Bp Bn];
    f = alpha*(K(id_test,SV(:))');
    f = f'-bbest;
    Y_te = Y(id_test);
    [ AUC_test(t) Accuracy_test(t) ] = Evaluation_AUC( f,Y(id_test) )
   
    if (mod(t,t_tick)==0)
        mistakes = [mistakes err_count/t];
        mistakes_idx = [mistakes_idx t];
        SVs = [SVs length(SV)];
        TMs=[TMs run_time];
    end
end
run_time = toc;
% figure;
% plot(mistakes_idx,mistakes);title('mistake rate');
% figure;
% plot(rt);title('Ap/An rate');
% % figure;
% % plot(mistakes_idx,SVs);title('SVs');
% figure;
% plot(AUC_test);title('AUC');
% figure;
% plot(Accuracy_test);title('Acc');
% figure;
% hist(App,20);
% % plot(App);title('App');
% figure;
% hist(Ann,20);
% % plot(Ann);title('Ann');
% % figure;
% % plot(a(a>0));title('a>0');
% % figure;
% % plot(a(a<0));title('a<0');
% figure;
% plot(rtt);
% [ma midx]=max(AUC_test)
%close all

alpha = [Ap An];
SV = [Bp Bn];

Model.b1 = nUpdateB(Ap,Bp,An,Bn,K);
f = alpha*(K(id_test,SV(:))');
f = f'-Model.b1;
[ AUC Accuracy ] = Evaluation_AUC( f,Y(id_test) );

Model.K = K;
Model.options = options;
Model.alpha = [App Ann];
Model.alpha1 = [Ap An];
Model.SV = SV;
Model.b = nUpdateB(App,Bp,Ann,Bn,K);
Model.b1 = nUpdateB(Ap,Bp,An,Bn,K);
AUC_list = AUC_test(mistakes_idx);
Acc_list = Accuracy_test(mistakes_idx);
fprintf(1,'The number of mistakes = %d\n', err_count);
end

