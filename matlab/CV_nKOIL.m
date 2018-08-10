function [Best_C, Best_sigma, Best_K ,Best_AUC] = CV_nKOIL(X,Y,ID_Train,options,Indices);
% Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% CV_KOIL: Cross Validation for KOIL
% -------------------------------------------------------------------------
% Input:
%        X_train:    the training instances
%        Y_train:    the vector of lables for X_train
%        Indices:    the indices for cross validation
%Output:
%         Best_C:    the best value of paramter C for OAMgra by Cross
%         Validation
% =========================================================================
cv_options = options;
Best_AUC=0;
Best_C=cv_options.C_KOIL;
Best_sigma = 2;

Y_train = Y(ID_Train);

load(strcat(cv_options.load_K,'_',num2str(0.25),'.mat'),'K');
Best_K=K;

%% for best sigma
rang3=[-5:1:5];
mean3_AUC = zeros(size(rang3,2),1);
mean3_Acc = zeros(size(rang3,2),1);
for t=1:size(rang3,2)
    i=rang3(t);
    cv_options.sigma=2^i;
  
    %% load the kernel matrix
    load(strcat(cv_options.load_K,'_',num2str(2^i),'.mat'),'K');
    %% run experiments:
    AUC_KOIL = zeros(5,1);
    Acc_KOIL = zeros(5,1);
    for j=1:5,
        id_train = ID_Train(Indices~=j);
        id_test  = ID_Train(Indices==j);
        Y_tr = Y_train(Indices~=j);
        Y_te = Y_train(Indices==j);
        [Model, AUC_KOIL(j), Acc_KOIL(j)] = nKOIL_RSC( id_train, id_test, Y, K, cv_options);        
    end
    
    mean3_AUC(t) = mean(AUC_KOIL);
    mean3_Acc(t) = mean(Acc_KOIL);
    if mean3_AUC(t)>Best_AUC,
        Best_AUC=mean3_AUC(t);
        Best_sigma = 2^i;
        Best_K = K;
    end
end
cv_options.sigma = Best_sigma;
% F1=figure,plot(rang3,mean3_AUC); 
% saveas(F1,strcat(options.save_path,'AUC_sigma.fig'));
% F2=figure,plot(rang3,mean3_Acc);
% saveas(F2,strcat(options.save_path,'Acc_sigma.fig'));


% for best C
rang3=[4:16];
for q=1:size(rang3,2)
    i=rang3(q);
    cv_options.C_KOIL=2^i;
   
%     fprintf(1,'i=%.f   C=%.f in CV_KOIL\n',i,2^i);
    
    %% run experiments:
    AUC_KOIL = zeros(5,1);
    Acc_KOIL = zeros(5,1);
    %% run experiments:
    for j=1:5,
        id_train = ID_Train(Indices~=j);
        id_test  = ID_Train(Indices==j);
        Y_tr = Y_train(Indices~=j);
        Y_te = Y_train(Indices==j);
        [Model, AUC_KOIL(j), Acc_KOIL(j)] = nKOIL_RSC( id_train, id_test, Y, Best_K, cv_options);    
    end
    
    mean_Acc(q) = mean(Acc_KOIL);
    mean_AUC(q) = mean(AUC_KOIL);
    if mean_AUC(q)>Best_AUC,
        Best_AUC=mean_AUC(q);
        Best_C=cv_options.C_KOIL;
    end
end
cv_options.C_KOIL = Best_C;

fprintf(1,'CV_KOIL: The best AUC value is %d\n', Best_AUC);
fprintf(1,'CV_KOIL: The best cost parameter is %d,\n', Best_C);
fprintf(1,'CV_KOIL: The best sigma parameter is %d,\n', Best_sigma);
save(strcat(options.save_path,'_options.mat'),'options')

F3=figure,plot(rang3,mean_AUC); 
saveas(F3,strcat(options.save_path,'AUC_C.fig'));
F4=figure,plot(rang3,mean_Acc);
saveas(F4,strcat(options.save_path,'Acc_C.fig'));
close all;

fprintf(1,'CV_KOIL: The best AUC value is %d\n', Best_AUC);
fprintf(1,'CV_KOIL: The best cost parameter is %d,\n', Best_C);
fprintf(1,'CV_KOIL: The best sigma parameter is %d,\n', Best_sigma);
save(strcat(options.save_path,'_KOIL_RSC_options.mat'),'options','Best_AUC');
end
