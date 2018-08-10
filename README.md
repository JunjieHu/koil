Kernelized Online Imbalanced Learning with Fixed Buddget
===
Implemented by [Junjie Hu](http://www.cs.cmu.edu/~junjieh/)

Contact: junjieh@cs.cmu.edu

If you use the codes in this repo, please cite our [AAAI paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9578).

	@inproceedings{hu2015kernelized,
	  title={Kernelized online imbalanced learning with fixed budgets},
	  author={Hu, Junjie and Yang, Haiqin and King, Irwin and Lyu, Michael R and So, Anthony Man-Cho},
	  booktitle={Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence},
	  pages={2666--2672},
	  year={2015},
	  organization={AAAI Press}
	}
	
If you use the codes for multi-kernel learning, please cite our [TNNLS paper](https://ieeexplore.ieee.org/document/7835710/).
	
	@article{hu2018online,
	  title={Online nonlinear AUC maximization for imbalanced data sets},
	  author={Hu, Junjie and Yang, Haiqin and Lyu, Michael R and King, Irwin and So, Anthony Man-Cho},
	  journal={IEEE transactions on neural networks and learning systems},
	  volume={29},
	  number={4},
	  pages={882--895},
	  year={2018},
	  publisher={IEEE}
	}
	

C++ Implementation
==
Brief: This is the C++ version KOIL. This program depends on the boost library.
For boost installation, please refer to [http://www.boost.org/](http://www.boost.org/).
This program is tested on Ubuntu 12.04 and 13.10. Open the terminal, redirect to the cpp folder and install by cmake.

	cd cpp
	cmake .
	make

Five executable functions are generated.

1. Online evaluation:

		./online [dataset_file] [C] [gamma] [loss type]
	Example to perform 20-runs KOIL based l2 loss, with C=1 and gamma=0.25:

		./online diabetes 1 0.25 l2


2. CV: cross validation by exploniential step
		
		./CV [dataset_file] [Num of clist] [Num of glist] [c start] [g start] [cstep] [gstep] [cvfold] [loss type]
	Example to perform 5-fold cross validation of KOIL based on l1 loss on diabetes dataset, where C in range [2^(-5:1:4)], gamma in range [2^(-10:1:9)].

		./CV diabetes 10 20 -5 -10 2 2 5 l1

 
3. CVP: cross validation by additional step

		./CVP [dataset_file] [Num of clist] [Num of glist] [c start] [g start] [cstep] [gstep] [cvfold] [loss type]
	Example to perform 5-fold cross validation of KOIL based on l1 loss on diabetes dataset, where C in range [1:10:91], gamma in range [0.01:0.05:0.901].
		
		./CVP diabetes 10 20 1 0.001 10 0.05 5 l1

 
4. CVM: cross validation by exploniential step for MKL
		
		./CVM [dataset_file] [Num of clist] [c start] [cstep] [cvfold] [loss type]
	Example to perform 5-fold cross validation of KOIL with MKL based on l1 loss on diabetes dataset, where C in range [2^(-10:1:9)]
		
		./CVM diabetes 10 20 -10 2 5 l1


5. mkl: KOIL with MKL
		
		./mkl [dataset_file] [loss type] [delta] [C] [degree num] [degree list]  [gamma list]
	Example to perform 20-runs KOIL with MKL based on l1 loss on diabetes, with 3 Polynomial Kernel where d={1,2,3} and 6 Gaussian Kernel where gamma={10^[-3:1:2]} 
		
		./mkl diabetes l1 0.5 1 3 1 2 3 0.001 0.01 0.1 1 10 100


Matlab Implementation
==

Brief: This is the Matlab implementation of KOIL. Open the Matlab and run the following three scripts in Matlab. You need to prepare the data and the kernel matrix in the same format as those in the data folder.

1. Test the performance of KOIL RS/RS++, FIFO/FIFO++, Infinite_Buffer

		run_demo.m

2. Test the effect of k

		run_demo_Ksize.m

3. Test the effect of buffer size

		run_demo_BufferSize.m

4. Test the 
		
		run_mkl_demo.m

