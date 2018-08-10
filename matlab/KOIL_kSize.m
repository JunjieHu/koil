function KOIL_kSize( Info )
% Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% Test the effect of different k values
% ——————————————————————————————————————————————————

%% load dataset
load(Info.load_path);
[n,d]       = size(data);
save_path = Info.save_path;
% set parameters
m           = n;

%% set parameters:
options.eta = 0.01           % learning rate for OLBBK
options.C_KOIL = 9;         % penalty parameter
options.sigma = 0.25;           % 'sigma'( kernel width)
options.Num_n = 100;
options.Num_p = 100;

%options.C=1;
options.k1=10;
options.k2=10;

options.save_path = Info.save_path;
options.load_K = Info.load_K;
load(strcat(options.load_K,'_',num2str(options.sigma),'.mat'),'K');
K_KOIL=K;
options.t_tick=round(n/15);  %'t_tick'(step size for plotting figures)



iter=10;
Varied_RS=zeros(20,iter);
Varied_RSP=zeros(20,iter);
Varied_FIFO=zeros(20,iter);
Varied_k=zeros(20,iter);

Num_buf = [1 10:10:options.Num_p];
for t=1:size(Num_buf,2)
    
    N=Num_buf(t);
    
    options.k1=N;
    options.k2=N;
    
    %%  choose a subset of the whole dataset (default: using all)
    Y = data(1:m,1);
    X = data(1:m,2:d);
    
    
    %% run experiments:
    for i=1:4,
        Indices = IdxStr.IdxCv(:,i);                  % Indices for cross     
        head=1;
        for j=1:5,          
            k=5*(i-1)+j;
            fprintf('The %d-th trial \n',k);
            ID_Train = find(Indices~=j);
            ID_Test  = find(Indices==j);
            Y_train = Y(Indices~=j,:);
            Y_test  = Y(Indices==j,:);            
            
            %% search the best parameters by cross validation
            length_j=sum(Indices~=j);
            tail=head+length_j-1;
            idx_j= IdxStr.IdxAsso(head:tail,i);
            head=1+tail;           
            if i==1&&j==1           
                tnum = floor(length(ID_Train)/options.t_tick);
            end
            
            %% online learning
            %1. KOIL_RSC
            [Model, AUC_KOIL_RSC(k), Acc_KOIL_RSC(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_RSC( ID_Train, ID_Test, Y, K_KOIL, options);
            KOIL_RSC(k,1)=AUC_KOIL_RSC(k);   
            
             %3. KOIL FIFOC
            [Model, AUC_KOIL_FIFOC(k), Acc_KOIL_FIFOC(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_FIFOC( ID_Train, ID_Test, Y, K_KOIL, options);
            KOIL_FIFOC(k,1)=AUC_KOIL_FIFOC(k);
            
        end
    end

    Varied_RSC(:,t)=KOIL_RSC;
    Varied_FIFOC(:,t)=KOIL_FIFOC;
    
end
save(strcat(save_path,'_kSize_KOIL_RSC_FIFOC.mat'));



F5=figure
mean_RSC = mean(Varied_RSC);
std_RSC = std(Varied_RSC);
errorbar(Num_buf, mean_RSC,std_RSC, 'r-o');
hold on
mean_FIFOC = mean(Varied_FIFOC);
std_FIFOC = std(Varied_FIFOC);
errorbar(Num_buf, mean_FIFOC,std_FIFOC,'b-x');
legend('KOIL_{RSC}','KOIL_{FIFOC}','Location','SouthEast');
xlabel('The buffer size');
ylabel('Average AUC value')
grid
saveas(F5,strcat(save_path,'_RSC_FIFOC_kSize.png'));
close all

end
