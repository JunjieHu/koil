function [ AUC Accuracy ] = Evaluation_AUC( f,Y_test )
% Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
% —————————————————————————
% Input:
% f: predicted value
% Y_test: true label (+/-1)
% Output: 
% AUC, Accuracy
% =========================
Accuracy=sum(sign(f)==Y_test)/length(Y_test);
correct=0;
for i=1:length(Y_test),
    for j=i+1:length(Y_test),
        if (Y_test(i)-Y_test(j))*(f(i)-f(j))>0,
            correct=correct+1;
        end
    end
end

Num_test_pos=sum(Y_test>0);
Num_test_neg=sum(Y_test<0);
AUC=correct/(Num_test_pos*Num_test_neg);
end

