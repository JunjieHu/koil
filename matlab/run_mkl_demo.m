function run_mkl_demo
% KOIL: Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% demo to test KOIL_RS++, KOIL_FIFO++, KOIL_RS, KOIL_FIFO, KOIL with infinite buffer

dataset={'glass'}      
      
n=size(dataset,1);

for i=1:1:n
    t=cellstr(dataset(i));
    st=char(t);
    Info.load_path=strcat('real_data/',st,'.mat');
    Info.save_path=strcat('result/KOIL_RSC_FIFOC_',st);
    Info.load_K=strcat('real_data/',st,'kernel');
%     Info.load_options = strcat('result/KOIL_options_UB2/',st,'_options_KOIL.mat');
     KOIL_RSC_evaluation(Info);

end



end
