function KOIL_BufferSize( Info )
% Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% Test the effect of different buffer sizes
% ——————————————————————————————————————————————————

%% load dataset
load(Info.load_path);
[n,d]       = size(data);
% set parameters
m           = n;

%% set parameters:
options.eta = 0.01           % learning rate for KOIL
options.C_KOIL = 9;         % penalty parameter
options.sigma = 0.25;           % 'sigma'( kernel width)
options.Num_n = 100;
options.Num_p = 100;

options.k1=10;
options.k2=10;

options.save_path = Info.save_path;
options.load_K = Info.load_K;
load(strcat(options.load_K,'_',num2str(options.sigma),'.mat'),'K');
K_KOIL=K;
options.t_tick=round(n/15);  %'t_tick'(step size for plotting figures)

%%
iter=8;
Varied_RSC=zeros(20,iter);
Varied_FIFOC=zeros(20,iter);


step = floor(0.8*n/iter);
N=4;
Num_buf = N+([1:iter]-1)*step;

for t=1:iter
    N= Num_buf(t);
    
    options.Num_n = N;
    options.Num_p = N;
    options.k1=ceil(N/10);
    options.k2=ceil(N/10);
    
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
            
             %3. KOIL FIFO
            [Model, AUC_KOIL_FIFOC(k), Acc_KOIL_FIFOC(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_FIFOC( ID_Train, ID_Test, Y, K_KOIL, options);
            KOIL_FIFOC(k,1)=AUC_KOIL_FIFOC(k);
            
        end
    end

    Varied_RSC(:,t)=KOIL_RSC;
    Varied_FIFOC(:,t)=KOIL_FIFOC;
    
end
save(strcat(Info.save_path,'_BufferSize_KOIL_RSC_FIFOC.mat'));



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
saveas(F5,strcat(Info.save_path,'_RSC_FIFOC_BufferSize.fig'));
close all

end
