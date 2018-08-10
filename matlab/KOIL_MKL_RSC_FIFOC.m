function KOIL_MKL_RSC_FIFOC( Info )
% Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% KOIL_RS++ and KOIL_FIFO++
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load dataset
load(Info.load_path);
[n,d]       = size(data);
% set parameters
m           = n;

% %% load options
% load(Info.load_options);
% load(strcat(options.load_K,'_',num2str(options.sigma),'.mat'),'K');
% K_KOIL=K;


%% set parameters:
options.eta = 0.01           % learning rate for KOIL
options.C_KOIL     = 5;           % penalty parameter
options.sigma = 2;           % 'sigma'( kernel width)
options.Num_n = 100;
options.Num_p = 100;
options.sigmalist=[2,3,4,5];
options.beta = 0.8;


options.t_tick=round(n/15);  %'t_tick'(step size for plotting figures)
options.save_path = Info.save_path;
options.load_K = Info.load_K;

%options.C=1;
options.k1=5;
options.k2=5;


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
%             [Best_C, Best_sigma, K_KOIL] = CV_nKOIL(X,Y,ID_Train,options,idx_j);
%             options.C_KOIL = Best_C;
%             options.sigma = Best_sigma;       

%             load(Info.load_options);
%             load(strcat(options.load_K,'_',num2str(options.sigma),'.mat'),'K');
            options.C_KOIL=7;
            options.sigma = 0.5;
            load(strcat(options.load_K,'_',num2str(options.sigma),'.mat'),'K');            
            K_KOIL=K;
            
            tnum = floor(length(ID_Train)/options.t_tick);
        end
        
        %% online learning
        %1. KOIL_RSC
        [Model_RSC, AUC_KOIL_RSC(k), Acc_KOIL_RSC(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_MKL_RSC( ID_Train, ID_Test, Y, K_KOIL, options);       
        AUC_list_KOIL_RSC(k,1:tnum)=AUC_list(1:tnum);
        Acc_list_KOIL_RSC(k,1:tnum)=Acc_list(1:tnum);
        nSV_KOIL_RSC(k) = length(Model_RSC.SV);
        err_KOIL_RSC(k) = err_count;
        time_KOIL_RSC(k) = run_time;
        mistakes_list_KOIL_RSC(k,1:tnum) = mistakes(1:tnum);
        SVs_KOIL_RSC(k,1:tnum) = SVs(1:tnum);
        TMs_KOIL_RSC(k,1:tnum) = TMs(1:tnum);
        AUC_KOIL_RSC(k);
        Acc_KOIL_RSC(k);
%         %2. KOIL_RS
%         [Model_RS, AUC_KOIL_RS(k), Acc_KOIL_RS(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_RS( ID_Train, ID_Test, Y, K_KOIL, options);
%         AUC_list_KOIL_RS(k,1:tnum)=AUC_list(1:tnum);
%         Acc_list_KOIL_RS(k,1:tnum)=Acc_list(1:tnum);
%         nSV_KOIL_RS(k) = length(Model_RS.SV);
%         err_KOIL_RS(k) = err_count;
%         time_KOIL_RS(k) = run_time;
%         mistakes_list_KOIL_RS(k,1:tnum) = mistakes(1:tnum);
%         SVs_KOIL_RS(k,1:tnum) = SVs(1:tnum);
%         TMs_KOIL_RS(k,1:tnum) = TMs(1:tnum);
        
%         %3. KOIL FIFOc
%         [Model_FIFOC, AUC_KOIL_FIFOC(k), Acc_KOIL_FIFOC(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_FIFOC( ID_Train, ID_Test, Y, K_KOIL, options);
%         AUC_list_KOIL_FIFOC(k,1:tnum)=AUC_list(1:tnum);
%         Acc_list_KOIL_FIFOC(k,1:tnum)=Acc_list(1:tnum);
%         nSV_KOIL_FIFOC(k) = length(Model_FIFOC.SV);
%         err_KOIL_FIFOC(k) = err_count;
%         time_KOIL_FIFOC(k) = run_time;
%         mistakes_list_KOIL_FIFOC(k,1:tnum) = mistakes(1:tnum);
%         SVs_KOIL_FIFOC(k,1:tnum) = SVs(1:tnum);
%         TMs_KOIL_FIFOC(k,1:tnum) = TMs(1:tnum);
%         AUC_KOIL_FIFOC(k);
%         Acc_KOIL_FIFOC(k);
        
%         %4. KOIL FIFO
%         [Model_FIFO, AUC_KOIL_FIFO(k), Acc_KOIL_FIFO(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_FIFO( ID_Train, ID_Test, Y, K_KOIL, options);
%         AUC_list_KOIL_FIFO(k,1:tnum)=AUC_list(1:tnum);
%         Acc_list_KOIL_FIFO(k,1:tnum)=Acc_list(1:tnum);
%         nSV_KOIL_FIFO(k) = length(Model_FIFO.SV);
%         err_KOIL_FIFO(k) = err_count;
%         time_KOIL_FIFO(k) = run_time;
%         mistakes_list_KOIL_FIFO(k,1:tnum) = mistakes(1:tnum);
%         SVs_KOIL_FIFO(k,1:tnum) = SVs(1:tnum);
%         TMs_KOIL_FIFO(k,1:tnum) = TMs(1:tnum);
%    
%         %5. KOIL_infinite buffer
%         inf_options = options;
%         inf_options.Num_n = 10000;
%         inf_options.Num_p = 10000;
%         [Model_KOIL, AUC_KOIL(k), Acc_KOIL(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_RSC( ID_Train, ID_Test, Y, K_KOIL, inf_options);       
%         AUC_list_KOIL(k,1:tnum)=AUC_list(1:tnum);
%         Acc_list_KOIL(k,1:tnum)=Acc_list(1:tnum);
%         nSV_KOIL(k) = length(Model_KOIL.SV);
%         err_KOIL(k) = err_count;
%         time_KOIL(k) = run_time;
%         mistakes_list_KOIL(k,1:tnum) = mistakes(1:tnum);
%         SVs_KOIL(k,1:tnum) = SVs(1:tnum);
%         TMs_KOIL(k,1:tnum) = TMs(1:tnum);
    end
end
save(strcat(Info.save_path,'_result_MKL_KOIL.mat'));

fid=fopen(strcat(Info.save_path,'_MKL_KOIL.txt'),'a+');

fprintf(fid,Info.save_path);
fprintf(fid,'\n-------------------------------------------------------------------------------\n');
fprintf(fid,'KOIL_RSC: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_KOIL_RSC), std(err_KOIL_RSC), mean(nSV_KOIL_RSC), std(nSV_KOIL_RSC), mean(time_KOIL_RSC), std(time_KOIL_RSC));
% fprintf(fid,'KOIL_RS: (number of mistakes, size of support vectors, cpu running time)\n');
% fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_KOIL_RS), std(err_KOIL_RS), mean(nSV_KOIL_RS), std(nSV_KOIL_RS), mean(time_KOIL_RS), std(time_KOIL_RS));
% fprintf(fid,'KOIL_FIFOC: (number of mistakes, size of support vectors, cpu running time)\n');
% fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_KOIL_FIFOC), std(err_KOIL_FIFOC), mean(nSV_KOIL_FIFOC), std(nSV_KOIL_FIFOC), mean(time_KOIL_FIFOC), std(time_KOIL_FIFOC));
% fprintf(fid,'KOIL_FIFO: (number of mistakes, size of support vectors, cpu running time)\n');
% fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_KOIL_FIFO), std(err_KOIL_FIFO), mean(nSV_KOIL_FIFO), std(nSV_KOIL_FIFO), mean(time_KOIL_FIFO), std(time_KOIL_FIFO));
% fprintf(fid,'KOIL_inf: (number of mistakes, size of support vectors, cpu running time)\n');
% fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_KOIL), std(err_KOIL), mean(nSV_KOIL), std(nSV_KOIL), mean(time_KOIL), std(time_KOIL));
fprintf(fid,'-------------------------------------------------------------------------------\n\n');


fprintf(fid,'\n-------------------------------------------------------------------------------\n');
fprintf(fid,'KOIL_RSC  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL_RSC),std(AUC_KOIL_RSC),mean(Acc_KOIL_RSC),std(Acc_KOIL_RSC),mean(time_KOIL_RSC));
% fprintf(fid,'KOIL_RS  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL_RS),std(AUC_KOIL_RS),mean(Acc_KOIL_RS),std(Acc_KOIL_RS),mean(time_KOIL_RS));
% fprintf(fid,'KOIL_FIFOC  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL_FIFOC),std(AUC_KOIL_FIFOC),mean(Acc_KOIL_FIFOC),std(Acc_KOIL_FIFOC),mean(time_KOIL_FIFOC));
% fprintf(fid,'KOIL_FIFO  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL_FIFO),std(AUC_KOIL_FIFO),mean(Acc_KOIL_FIFO),std(Acc_KOIL_FIFO),mean(time_KOIL_FIFO));
% fprintf(fid,'KOIL_inf  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL),std(AUC_KOIL),mean(Acc_KOIL),std(Acc_KOIL),mean(time_KOIL));
fprintf(fid,'-------------------------------------------------------------------------------\n');
fclose(fid);

fprintf(1,Info.save_path);
fprintf(1,'\n-------------------------------------------------------------------------------\n');
fprintf(1,'KOIL_RSC  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL_RSC),std(AUC_KOIL_RSC),mean(Acc_KOIL_RSC),std(Acc_KOIL_RSC),mean(time_KOIL_RSC));
% fprintf(1,'KOIL_RS  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL_RS),std(AUC_KOIL_RS),mean(Acc_KOIL_RS),std(Acc_KOIL_RS),mean(time_KOIL_RS));
% fprintf(1,'KOIL_FIFOC  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL_FIFOC),std(AUC_KOIL_FIFOC),mean(Acc_KOIL_FIFOC),std(Acc_KOIL_FIFOC),mean(time_KOIL_FIFOC));
% fprintf(1,'KOIL_FIFO  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL_FIFO),std(AUC_KOIL_FIFO),mean(Acc_KOIL_FIFO),std(Acc_KOIL_FIFO),mean(time_KOIL_FIFO));
% fprintf(1,'KOIL_inf  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_KOIL),std(AUC_KOIL),mean(Acc_KOIL),std(Acc_KOIL),mean(time_KOIL));
fprintf(1,'-------------------------------------------------------------------------------\n');

end
