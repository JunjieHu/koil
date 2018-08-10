function run_demo_BufferSize
% KOIL: Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% demo to test the buffer size

dataset={'glass'}      
      
n=size(dataset,1);

for i=1:1:n
    t=cellstr(dataset(i));
    st=char(t);
    Info.load_path=strcat('real_data/',st,'.mat');
    Info.save_path=strcat('result/KOIL',st);
    Info.load_K=strcat('real_data/',st,'kernel');

    KOIL_BufferSize(Info);

end



end
