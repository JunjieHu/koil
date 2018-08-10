function [ bbest accbest ] = UpdateB( Ap,Bp,An,Bn,K )
% KOIL: Kernelized Online Imbalanced Learning with Fixed Buddget (AAAI 2015)
% Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
% Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
 

pnum = size(Bp,2);
nnum = size(Bn,2);
y=[ones(pnum,1);zeros(nnum,1)];

%% test performance
alpha = [Ap An];
SV = [Bp Bn];
%% for test samples
temp = K(SV,SV(:))';
f_y = (alpha*(temp))';

if(isempty(Bp)||isempty(Bp))
    bbest = 0;
    accbest = 0;
    return;
end

pmin = min(f_y(1:pnum,1));
nmax = max(f_y(pnum+1:end,1));

if(pmin>=nmax)
    bbest = (pmin+nmax)/2;
    accbest = 1;
else
    bstep=(nmax-pmin)/500;
    bbest = 0;
    accbest = 0;
    for cb=pmin:bstep:nmax
        label = (f_y-cb)>=0;
        acc = sum(label==y)/(pnum+nnum);
        if(acc>accbest)
            bbest = cb;
            accbest = acc;
        end        
    end
end

