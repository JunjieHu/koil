function KOIL_RSC_evaluation( Info )
% Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% Evaluate the KOIL_RS++

%% load dataset
load(Info.load_path);
[n,d]       = size(data);
% set parameters
m           = n;

%% set parameters:
options.eta = 0.01;           % learning rate for OLBBK
options.C_KOIL = 5;           % penalty parameter
options.sigma = 2;            % 'sigma'( kernel width)
options.Num_n = 100;
options.Num_p = 100;


options.t_tick=round(n/15);  %'t_tick'(step size for plotting figures)
options.save_path = Info.save_path;
options.load_K = Info.load_K;

%options.C=1;
options.k1=10;
options.k2=10;


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
%             [Best_C, Best_sigma, K_OLBBK] = CV_nOLBBK(X,Y,ID_Train,options,idx_j);
%             options.C_OLBBK = Best_C;
%             options.sigma = Best_sigma;       

%             load(Info.load_options);
%             load(strcat(options.load_K,'_',num2str(options.sigma),'.mat'),'K');
            options.C_OLBBK=2;
            options.sigma = 0.25;
            load(strcat(options.load_K,'_',num2str(0.25),'.mat'),'K');            
            K_OLBBK=K;
            
            tnum = floor(length(ID_Train)/options.t_tick);
        end
        
        %% online learning
        %1. KOIL_RSC
        [Model_RSC, AUC_OLBBK_RSC(k), Acc_OLBBK_RSC(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_RSC( ID_Train, ID_Test, Y, K_OLBBK, options);       
        AUC_list_OLBBK_RSC(k,1:tnum)=AUC_list(1:tnum);
        Acc_list_OLBBK_RSC(k,1:tnum)=Acc_list(1:tnum);
        nSV_OLBBK_RSC(k) = length(Model_RSC.SV);
        err_OLBBK_RSC(k) = err_count;
        time_OLBBK_RSC(k) = run_time;
        mistakes_list_OLBBK_RSC(k,1:tnum) = mistakes(1:tnum);
        SVs_OLBBK_RSC(k,1:tnum) = SVs(1:tnum);
        TMs_OLBBK_RSC(k,1:tnum) = TMs(1:tnum);
        
        %2. KOIL_RS
        [Model_RS, AUC_OLBBK_RS(k), Acc_OLBBK_RS(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_RS( ID_Train, ID_Test, Y, K_OLBBK, options);
        AUC_list_OLBBK_RS(k,1:tnum)=AUC_list(1:tnum);
        Acc_list_OLBBK_RS(k,1:tnum)=Acc_list(1:tnum);
        nSV_OLBBK_RS(k) = length(Model_RS.SV);
        err_OLBBK_RS(k) = err_count;
        time_OLBBK_RS(k) = run_time;
        mistakes_list_OLBBK_RS(k,1:tnum) = mistakes(1:tnum);
        SVs_OLBBK_RS(k,1:tnum) = SVs(1:tnum);
        TMs_OLBBK_RS(k,1:tnum) = TMs(1:tnum);
        
        %3. KOIL FIFOc
        [Model_FIFOC, AUC_OLBBK_FIFOC(k), Acc_OLBBK_FIFOC(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_FIFOC( ID_Train, ID_Test, Y, K_OLBBK, options);
        AUC_list_OLBBK_FIFOC(k,1:tnum)=AUC_list(1:tnum);
        Acc_list_OLBBK_FIFOC(k,1:tnum)=Acc_list(1:tnum);
        nSV_OLBBK_FIFOC(k) = length(Model_FIFOC.SV);
        err_OLBBK_FIFOC(k) = err_count;
        time_OLBBK_FIFOC(k) = run_time;
        mistakes_list_OLBBK_FIFOC(k,1:tnum) = mistakes(1:tnum);
        SVs_OLBBK_FIFOC(k,1:tnum) = SVs(1:tnum);
        TMs_OLBBK_FIFOC(k,1:tnum) = TMs(1:tnum);
        
        %4. KOIL FIFO
        [Model_FIFO, AUC_OLBBK_FIFO(k), Acc_OLBBK_FIFO(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_FIFO( ID_Train, ID_Test, Y, K_OLBBK, options);
        AUC_list_OLBBK_FIFO(k,1:tnum)=AUC_list(1:tnum);
        Acc_list_OLBBK_FIFO(k,1:tnum)=Acc_list(1:tnum);
        nSV_OLBBK_FIFO(k) = length(Model_FIFO.SV);
        err_OLBBK_FIFO(k) = err_count;
        time_OLBBK_FIFO(k) = run_time;
        mistakes_list_OLBBK_FIFO(k,1:tnum) = mistakes(1:tnum);
        SVs_OLBBK_FIFO(k,1:tnum) = SVs(1:tnum);
        TMs_OLBBK_FIFO(k,1:tnum) = TMs(1:tnum);
   
        %5. KOIL_infinite buffer
        inf_options = options;
        inf_options.Num_n = 10000;
        inf_options.Num_p = 10000;
        [Model_KOIL, AUC_OLBBK(k), Acc_OLBBK(k),err_count, run_time, mistakes, mistakes_idx, SVs, TMs,AUC_list,Acc_list] = nKOIL_RSC( ID_Train, ID_Test, Y, K_OLBBK, inf_options);       
        AUC_list_OLBBK(k,1:tnum)=AUC_list(1:tnum);
        Acc_list_OLBBK(k,1:tnum)=Acc_list(1:tnum);
        nSV_OLBBK(k) = length(Model_KOIL.SV);
        err_OLBBK(k) = err_count;
        time_OLBBK(k) = run_time;
        mistakes_list_OLBBK(k,1:tnum) = mistakes(1:tnum);
        SVs_OLBBK(k,1:tnum) = SVs(1:tnum);
        TMs_OLBBK(k,1:tnum) = TMs(1:tnum);
    end
end
save(strcat(Info.save_path,'_result_KOIL.mat'));

F1 = figure
mean_mistakes_OLBBK_RSC = mean(mistakes_list_OLBBK_RSC);
plot(mistakes_idx(1:tnum), mean_mistakes_OLBBK_RSC,'g-*');
hold on
mean_mistakes_OLBBK_RS = mean(mistakes_list_OLBBK_RS);
plot(mistakes_idx(1:tnum), mean_mistakes_OLBBK_RS,'r-x');
hold on
mean_mistakes_OLBBK_FIFOC = mean(mistakes_list_OLBBK_FIFOC);
plot(mistakes_idx(1:tnum), mean_mistakes_OLBBK_FIFOC,'b-o');
hold on
mean_mistakes_OLBBK_FIFO = mean(mistakes_list_OLBBK_FIFO);
plot(mistakes_idx(1:tnum), mean_mistakes_OLBBK_FIFO,'c-v');
hold on
mean_mistakes_OLBBK = mean(mistakes_list_OLBBK);
plot(mistakes_idx(1:tnum), mean_mistakes_OLBBK,'k-d');
legend('KOIL_{RSC}','KOIL_{RS}','KOIL_{FIFOC}','KOIL_{FIFO}','KOIL_{inf}','Location','SouthEast');
xlabel('Number of samples');
ylabel('Online average rate of mistakes')
grid
saveas(F1,strcat(Info.save_path,'F1.png'));

F2=figure
mean_SV_OLBBK_RSC = mean(SVs_OLBBK_RSC);
plot(mistakes_idx(1:tnum), mean_SV_OLBBK_RSC,'g-*');
hold on
mean_SV_OLBBK_RS = mean(SVs_OLBBK_RS);
plot(mistakes_idx(1:tnum), mean_SV_OLBBK_RS,'r-x');
hold on
mean_SV_OLBBK_FIFOC = mean(SVs_OLBBK_FIFOC);
plot(mistakes_idx(1:tnum), mean_SV_OLBBK_FIFOC,'b-o');
hold on
mean_SV_OLBBK_FIFO = mean(SVs_OLBBK_FIFO);
plot(mistakes_idx(1:tnum), mean_SV_OLBBK_FIFO,'c-v');
hold on
mean_SV_OLBBK = mean(SVs_OLBBK);
plot(mistakes_idx(1:tnum), mean_SV_OLBBK,'k-d');
legend('KOIL_{RSC}','KOIL_{RS}','KOIL_{FIFOC}','KOIL_{FIFO}','KOIL_{inf}','Location','SouthEast');
xlabel('Number of samples');
ylabel('Online average number of support vectors')
grid
saveas(F2,strcat(Info.save_path,'F2.png'));

F3=figure
mean_TM_OLBBK_RSC = log(mean(TMs_OLBBK_RSC))/log(10);
plot(mistakes_idx(1:tnum), mean_TM_OLBBK_RSC,'g-*');
hold on
mean_TM_OLBBK_RS = log(mean(TMs_OLBBK_RS))/log(10);
plot(mistakes_idx(1:tnum), mean_TM_OLBBK_RS,'r-x');
hold on
mean_TM_OLBBK_FIFOC = log(mean(TMs_OLBBK_FIFOC))/log(10);
plot(mistakes_idx(1:tnum), mean_TM_OLBBK_FIFOC,'b-o');
hold on
mean_TM_OLBBK_FIFO = log(mean(TMs_OLBBK_FIFO))/log(10);
plot(mistakes_idx(1:tnum), mean_TM_OLBBK_FIFO,'c-v');
hold on
mean_TM_OLBBK = log(mean(TMs_OLBBK))/log(10);
plot(mistakes_idx(1:tnum), mean_TM_OLBBK,'k-d');
legend('KOIL_{RSC}','KOIL_{RS}','KOIL_{FIFOC}','KOIL_{FIFO}','KOIL_{inf}','Location','SouthEast');
xlabel('Number of samples');
ylabel('average time cost (log_{10} t)')
grid
saveas(F3,strcat(Info.save_path,'F3.png'));

F4 = figure
mean_AUC_OLBBK_RSC = mean(AUC_list_OLBBK_RSC);
plot(mistakes_idx(1:tnum), mean_AUC_OLBBK_RSC,'g-*');
hold on
mean_AUC_OLBBK_RS = mean(AUC_list_OLBBK_RS);
plot(mistakes_idx(1:tnum), mean_AUC_OLBBK_RS,'r-x');
hold on
mean_AUC_OLBBK_FIFOC = mean(AUC_list_OLBBK_FIFOC);
plot(mistakes_idx(1:tnum), mean_AUC_OLBBK_FIFOC,'b-o');
hold on
mean_AUC_OLBBK_FIFO = mean(AUC_list_OLBBK_FIFO);
plot(mistakes_idx(1:tnum), mean_AUC_OLBBK_FIFO,'c-v');
hold on
mean_AUC_OLBBK = mean(AUC_list_OLBBK);
plot(mistakes_idx(1:tnum), mean_AUC_OLBBK,'k-d');
legend('KOIL_{RSC}','KOIL_{RS}','KOIL_{FIFOC}','KOIL_{FIFO}','KOIL_{inf}','Location','SouthEast');
xlabel('Number of samples');
ylabel('Online average rate of AUC');
grid
saveas(F4,strcat(Info.save_path,'F4.png'));

F5 = figure
mean_Acc_OLBBK_RSC = mean(Acc_list_OLBBK_RSC);
plot(mistakes_idx(1:tnum), mean_Acc_OLBBK_RSC,'g-*');
hold on
mean_Acc_OLBBK_RS = mean(Acc_list_OLBBK_RS);
plot(mistakes_idx(1:tnum), mean_Acc_OLBBK_RS,'r-x');
hold on
mean_Acc_OLBBK_FIFOC = mean(Acc_list_OLBBK_FIFOC);
plot(mistakes_idx(1:tnum), mean_Acc_OLBBK_FIFOC,'b-o');
hold on
mean_Acc_OLBBK_FIFO = mean(Acc_list_OLBBK_FIFO);
plot(mistakes_idx(1:tnum), mean_Acc_OLBBK_FIFO,'c-v');
hold on
mean_Acc_OLBBK = mean(Acc_list_OLBBK);
plot(mistakes_idx(1:tnum), mean_Acc_OLBBK,'k-d');
legend('KOIL_{RSC}','KOIL_{RS}','KOIL_{FIFOC}','KOIL_{FIFO}','KOIL_{inf}','Location','SouthEast');
xlabel('Number of samples');
ylabel('Online average rate of Accuracy');
grid
saveas(F5,strcat(Info.save_path,'F5.png'));

close all



fid=fopen(strcat(Info.save_path,'_evaluation_KOIL.txt'),'a+');

fprintf(fid,Info.save_path);
fprintf(fid,'\n-------------------------------------------------------------------------------\n');
fprintf(fid,'KOIL_RSC: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_OLBBK_RSC), std(err_OLBBK_RSC), mean(nSV_OLBBK_RSC), std(nSV_OLBBK_RSC), mean(time_OLBBK_RSC), std(time_OLBBK_RSC));
fprintf(fid,'KOIL_RS: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_OLBBK_RS), std(err_OLBBK_RS), mean(nSV_OLBBK_RS), std(nSV_OLBBK_RS), mean(time_OLBBK_RS), std(time_OLBBK_RS));
fprintf(fid,'KOIL_FIFOC: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_OLBBK_FIFOC), std(err_OLBBK_FIFOC), mean(nSV_OLBBK_FIFOC), std(nSV_OLBBK_FIFOC), mean(time_OLBBK_FIFOC), std(time_OLBBK_FIFOC));
fprintf(fid,'KOIL_FIFO: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_OLBBK_FIFO), std(err_OLBBK_FIFO), mean(nSV_OLBBK_FIFO), std(nSV_OLBBK_FIFO), mean(time_OLBBK_FIFO), std(time_OLBBK_FIFO));
fprintf(fid,'KOIL_inf: (number of mistakes, size of support vectors, cpu running time)\n');
fprintf(fid,'%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_OLBBK), std(err_OLBBK), mean(nSV_OLBBK), std(nSV_OLBBK), mean(time_OLBBK), std(time_OLBBK));
fprintf(fid,'-------------------------------------------------------------------------------\n\n');


fprintf(fid,'\n-------------------------------------------------------------------------------\n');
fprintf(fid,'KOIL_RSC  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK_RSC),std(AUC_OLBBK_RSC),mean(Acc_OLBBK_RSC),std(Acc_OLBBK_RSC),mean(time_OLBBK_RSC));
fprintf(fid,'KOIL_RS  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK_RS),std(AUC_OLBBK_RS),mean(Acc_OLBBK_RS),std(Acc_OLBBK_RS),mean(time_OLBBK_RS));
fprintf(fid,'KOIL_FIFOC  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK_FIFOC),std(AUC_OLBBK_FIFOC),mean(Acc_OLBBK_FIFOC),std(Acc_OLBBK_FIFOC),mean(time_OLBBK_FIFOC));
fprintf(fid,'KOIL_FIFO  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK_FIFO),std(AUC_OLBBK_FIFO),mean(Acc_OLBBK_FIFO),std(Acc_OLBBK_FIFO),mean(time_OLBBK_FIFO));
fprintf(fid,'KOIL_inf  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK),std(AUC_OLBBK),mean(Acc_OLBBK),std(Acc_OLBBK),mean(time_OLBBK));
fprintf(fid,'-------------------------------------------------------------------------------\n');
fclose(fid);

fprintf(1,Info.save_path);
fprintf(1,'\n-------------------------------------------------------------------------------\n');
fprintf(1,'KOIL_RSC  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK_RSC),std(AUC_OLBBK_RSC),mean(Acc_OLBBK_RSC),std(Acc_OLBBK_RSC),mean(time_OLBBK_RSC));
fprintf(1,'KOIL_RS  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK_RS),std(AUC_OLBBK_RS),mean(Acc_OLBBK_RS),std(Acc_OLBBK_RS),mean(time_OLBBK_RS));
fprintf(1,'KOIL_FIFOC  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK_FIFOC),std(AUC_OLBBK_FIFOC),mean(Acc_OLBBK_FIFOC),std(Acc_OLBBK_FIFOC),mean(time_OLBBK_FIFOC));
fprintf(1,'KOIL_FIFO  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK_FIFO),std(AUC_OLBBK_FIFO),mean(Acc_OLBBK_FIFO),std(Acc_OLBBK_FIFO),mean(time_OLBBK_FIFO));
fprintf(1,'KOIL_inf  &%.3f \t$\\pm$ %.3f \t &%.3f \t$\\pm$ %.3f \t & %.3f  \t \\\\\n', mean(AUC_OLBBK),std(AUC_OLBBK),mean(Acc_OLBBK),std(Acc_OLBBK),mean(time_OLBBK));
fprintf(1,'-------------------------------------------------------------------------------\n');

end
