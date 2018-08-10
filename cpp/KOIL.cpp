/* ****************************************************************
* Kernelized Online Imbalanced Learning with Fixed Buddget
* Junjie Hu, Haiqin Yang, Irwin King, Michael R. Lyu, Anthony Man-Cho So
* in Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence (AAAI 2015), Austin, Texas, USA, January 25–29, 2015.
* Implemented by Junjie Hu (jjhu@cse.cuhk.edu.hk)
******************************************************************** */

#include "KOIL.h"
#include "utility.h"
#include "svm.h"
#include <ctime>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#define debug 0
#define debug_cv 0
#define log 0

using namespace std;

/****************************************
 * KOIL: Main Functions
 * **************************************/
// KOIL_RS++
void KOIL::rs_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_model& model, double& AUC,double& Accuracy, double& time, int& err_count)
{
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    double* AUC_test = Malloc(double,cnt_train);
    double* Acc_test = Malloc(double,cnt_train);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    // initial prarameters
    int p,n,ridx;
    double at;
    clock_t start, end;
    bool flag;

    start = clock();
    err_count = 0;
    p=n=0;
    for(int t=0;t<cnt_train;t++){
        char buffer[100];
#if log
        sprintf(buffer, "The %d-th trial, Y(id)=%d",t+1,prob.y[id_train[t]]);
        write_log(buffer,0,string(buffer),string("result/Clog.txt"));
        //cout<<t<<"-th trial"<<endl;
#endif
        // get the xt and yt
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=model.predict(xt);
        if(ft*yt<=0)
            err_count++;

        if(yt==1)
        {
            p++;
            if(losstype=="l1")
                update_kernel(xt,yt,model,at);
            else update_kernel_l2(xt,yt,model,at);
            rs_update_budget(xt,at,model.max_pos_n,p,
                             model,model.pos_alpha,model.pos_SV,model.pos_n,flag,ridx);
        }else{
            n++;
            if(losstype=="l1")
                update_kernel(xt,yt,model,at);
            else update_kernel_l2(xt,yt,model,at);
            rs_update_budget(xt,at,model.max_neg_n,n,
                             model,model.neg_alpha,model.neg_SV,model.neg_n,flag,ridx);
        }

#if debug
        cout<<"rs pos_alpha:"<<endl;
        for(int i=0;i<model.pos_n;i++)
            cout<<model.pos_alpha[i]<<" ";
        cout<<endl;

        cout<<"rs neg_alpha:"<<endl;
        for(int i=0;i<model.neg_n;i++)
            cout<<model.neg_alpha[i]<<" ";
        cout<<endl<<endl;
#endif
        //cout<<"After update:flag = "<<flag<<", ridx="<<ridx<<", at = "<<at<<",pos_n="<<rs_model.pos_n<<endl<<endl;
        update_b(model);

#if debug
        cout<<"b = "<<model.b<<endl;
        cout<<endl;
        double* f_test = model.predict_list(x_test,cnt_test);
        evaluate_AUC(f_test,y_test,cnt_test,AUC_test[t],Acc_test[t]);
        cout<<t<<"-th trial:AUC="<<AUC_test[t]<<", Acc="<<Acc_test[t];
        cout<<endl;
#endif
#if log
        write_log(model.pos_alpha,model.pos_n,"Positive Ap","result/Clog.txt");
        write_log(model.neg_alpha,model.neg_n,"Negative Ap","result/Clog.txt");
        sprintf(buffer, "B: %.6f",model.b);
        write_log(buffer,0,string(buffer),string("result/Clog.txt"));
#endif
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

#if log
    write_log(AUC_test,cnt_train,"",this->log_file);
    write_log(Acc_test,cnt_train,"","result/acc.txt");
#endif

    // calculate AUC and accuracy on test set
    double* f_test = model.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_FIFO++
void KOIL::fifo_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_model& model,double& AUC,double& Accuracy, double& time, int& err_count)
{
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int p,n,ridx;
    double at;
    clock_t start, end;
    bool flag;

    start = clock();
    err_count = 0;
    p=n=0;
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=fifo_model.predict(xt);
        if(ft*yt<=0)
            err_count++;

        if(yt==1)
        {
            p++;
            if(losstype=="l1")
                update_kernel(xt,yt,model,at);
            else update_kernel_l2(xt,yt,model,at);
            fifo_update_budget(xt,at,model.max_pos_n,model,model.fpidx,
                               model.pos_alpha,model.pos_SV,model.pos_n,flag,ridx);
        }else{
            n++;
            if(losstype=="l1")
                update_kernel(xt,yt,model,at);
            else update_kernel_l2(xt,yt,model,at);
            fifo_update_budget(xt,at,model.max_neg_n,model,model.fnidx,
                               model.neg_alpha,model.neg_SV,model.neg_n,flag,ridx);
        }
        update_b(model);
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    double* f_test = model.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

/****************************************
 * KOIL: Helper Functions
 * **************************************/
/**
 * @brief update the weight for SV
 *
 * @param xt the t-th sample xt
 * @param yt the label of xt
 * @param model the current decision function f
 * @return at return the weight of xt
 */
void KOIL::update_kernel(svm_node* xt,double yt, svm_model& model, double& at)
{
    svm_node** same_sv;
    svm_node** oppo_sv;
    double* same_alpha;
    double* oppo_alpha;
    int same_n, oppo_n;

    if(yt == 1){
       same_sv = model.pos_SV;
       same_alpha = model.pos_alpha;
       same_n = model.pos_n;
       oppo_sv = model.neg_SV;
       oppo_alpha = model.neg_alpha;
       oppo_n = model.neg_n;
    }else{
        oppo_sv = model.pos_SV;
        oppo_alpha = model.pos_alpha;
        oppo_n = model.pos_n;
        same_sv = model.neg_SV;
        same_alpha = model.neg_alpha;
        same_n = model.neg_n;
    }

    // make prediction to xt
    double ft = model.predict(xt);
    double* fb = Malloc(double,oppo_n);
    vector<double> loss;
    vector<pair<double,int>> simpair;
    memset(fb,0,oppo_n);
    // find the k-nearest opposite SV which violates the pairwise loss
    for(int i=0;i<oppo_n;i++){
        if(oppo_alpha[i]==0)
            continue;
        fb[i] = model.predict(oppo_sv[i]);
        if(1>yt*(ft-fb[i])){
            loss.push_back(ft-fb[i]);
            simpair.push_back(pair<double,int> (model.kernel_func(xt,oppo_sv[i]),i));
        }
    }

#if debug
    cout<<"Print fxt:"<<ft+model.b<<endl;
    cout<<"Print fb:"<<endl;
    for(int i=0;i<oppo_n;i++){
        // test: print fb[i]
        cout<<fb[i]+model.b<<" ";
    }
    cout<<endl;
#endif

    // (1-eta)*alpha degrade with eta
    for(int i=0;i<oppo_n;i++)
        oppo_alpha[i] *=(1-model.param.eta);
    for(int i=0;i<same_n;i++)
        same_alpha[i] *=(1-model.param.eta);

    //Matlab
    if(oppo_n<1){
        at = model.param.C * model.param.eta * yt;
    }else{
        if(same_n<1 && simpair.size() == 0){
            at = model.param.C * model.param.eta * yt;
        }else{
            if(simpair.size()<model.k_num){
                at = simpair.size() * model.param.C * model.param.eta * yt;
                for(int i=0;i<simpair.size();i++)
                    oppo_alpha[simpair[i].second] -= model.param.C * model.param.eta * yt;
            }else{
                at = model.k_num * model.param.C * model.param.eta * yt;
                sort(simpair.begin(),simpair.end(),comparator<double>);
#if debug
                for(int i=0;i<simpair.size();i++){
                    cout<<simpair[i].first<<",order :"<<simpair[i].second<<endl;
                }
#endif
                for(int i=0;i<model.k_num;i++){
                    oppo_alpha[simpair[i].second] -= model.param.C * model.param.eta * yt;
                }
            }
        }
    }
}

/**
 * @brief update the weight for SV based on smooth pairwise hinge loss
 *
 * @param xt the t-th sample xt
 * @param yt the label of xt
 * @param model the current decision function f
 * @return at return the weight of xt
 */
void KOIL::update_kernel_l2(svm_node* xt,double yt, svm_model& model, double& at)
{
    svm_node** same_sv;
    svm_node** oppo_sv;
    double* same_alpha;
    double* oppo_alpha;
    int same_n, oppo_n;

    if(yt == 1){
       same_sv = model.pos_SV;
       same_alpha = model.pos_alpha;
       same_n = model.pos_n;
       oppo_sv = model.neg_SV;
       oppo_alpha = model.neg_alpha;
       oppo_n = model.neg_n;
    }else{
        oppo_sv = model.pos_SV;
        oppo_alpha = model.pos_alpha;
        oppo_n = model.pos_n;
        same_sv = model.neg_SV;
        same_alpha = model.neg_alpha;
        same_n = model.neg_n;
    }

    // make prediction to xt
    double ft = model.predict(xt);
    double* fb = Malloc(double,oppo_n);
    vector<double> loss;
    vector<pair<double,int>> simpair;
    memset(fb,0,oppo_n);
    // find the k-nearest opposite SV which violates the pairwise loss
    for(int i=0;i<oppo_n;i++){
        if(oppo_alpha[i]==0)
            continue;
        fb[i] = model.predict(oppo_sv[i]);
        if(1>yt*(ft-fb[i])){
            loss.push_back(ft-fb[i]);
            simpair.push_back(pair<double,int> (model.kernel_func(xt,oppo_sv[i]),i));
        }
    }

#if debug
    cout<<"Print fxt:"<<ft+model.b<<endl;
    cout<<"Print fb:"<<endl;
    for(int i=0;i<oppo_n;i++){
        // test: print fb[i]
        cout<<fb[i]+model.b<<" ";
    }
    cout<<endl;
#endif

    // (1-eta)*alpha degrade with eta
    for(int i=0;i<oppo_n;i++)
        oppo_alpha[i] *=(1-model.param.eta);
    for(int i=0;i<same_n;i++)
        same_alpha[i] *=(1-model.param.eta);

    //Matlab
    if(oppo_n<1){
        at = model.param.C * model.param.eta * yt;
    }else{
        if(same_n<1 && simpair.size() == 0){
            at = model.param.C * model.param.eta * yt;
        }else{
            if(simpair.size()<model.k_num){
                at = 0;
                for(int i=0;i<simpair.size();i++){
                    at += (ft-fb[simpair[i].second])*model.param.C*model.param.eta*yt;
                    oppo_alpha[simpair[i].second] -= model.param.C * model.param.eta * yt;
                }
            }else{
                at = 0;
                sort(simpair.begin(),simpair.end(),comparator<double>);
#if debug
                for(int i=0;i<simpair.size();i++){
                    cout<<simpair[i].first<<",order :"<<simpair[i].second<<endl;
                }
#endif
                for(int i=0;i<model.k_num;i++){
                    at += (ft-fb[simpair[i].second])*model.param.C*model.param.eta*yt;
                    oppo_alpha[simpair[i].second] -= model.param.C * model.param.eta * yt;
                }
            }
        }
    }
}


/**
 * @brief KOIL_RS++: update budget
 *
 * @param xt xt the t-th sample xt
 * @param at the weight of xt
 * @param max_n the maximun number for the buffer
 * @param t the current iteration
 * @param model the current decision function f
 * @param alpha the weights of SVs, which have the same label with xt
 * @param SV the SV, which have the same label with xt
 * @param cur_n current number of SVs in the buffer
 * @param flag indicate whether xt is put in the buffer or not
 * @param ridx the replaced index for xt if xt is put in the buffer
 */
void KOIL::rs_update_budget(svm_node* xt,double at,int max_n,int t,
                            svm_model& model, double* &alpha,svm_node** &SV,int& cur_n,bool& flag, int& ridx)
{
    //    cout<<"at="<<at<<",t="<<t<<endl;
    // variable initialization
    ridx = -1;
    flag = false;
    double ac = 0;
    svm_node* svc = NULL;

    if(at==0)
        return;
    if(cur_n<max_n)
    {
        alpha[cur_n] = at;
        SV[cur_n] = xt;
        flag = true;
        cur_n ++;
    }else{
        // replace one instance from SV with Probability = max_n/t
        //cout<<"t="<<t<<endl;
        srand((unsigned)time(0));
        //int tempind = rand()%t;
        //cout<<tempind;
        if(rand()%t<max_n){
            ridx = rand()%max_n;
            ac = alpha[ridx];
            svc = SV[ridx];
            alpha[ridx] = at;
            SV[ridx]=xt;
            flag = true;
        }else{
            ac = at;
            svc = xt;
        }

        // find the most similar SV for compensation
        int cidx = 0;
        double ma = 0;
        for(int i=0;i<max_n;i++){
            double temp = model.kernel_func(svc,SV[i]);
            if(temp>ma){
                cidx = i;
                ma = temp;
            }
        }
        alpha[cidx] += ac;
    }
}

/**
 * @brief KOIL_FIFO++: update budget
 *
 * @param xt xt the t-th sample xt
 * @param at the weight of xt
 * @param max_n the maximun number for the buffer
 * @param model the current decision function f
 * @param fidx the index of the first SV in the buffer (FIFO)
 * @param alpha the weights of SVs, which have the same label with xt
 * @param SV the SV, which have the same label with xt
 * @param cur_n current number of SVs in the buffer
 * @param flag indicate whether xt is put in the buffer or not
 * @param ridx the replaced index for xt if xt is put in the buffer
 */
void KOIL::fifo_update_budget(svm_node* xt,double at,int max_n,
                              svm_model& model,int& fidx,double* &alpha,svm_node** &SV,int& cur_n,bool& flag, int& ridx)
{
    // variable initialization
    ridx = -1;
    flag = false;
    double ac = 0;
    svm_node* svc = NULL;

    if(at==0)
        return;
    if(cur_n<max_n)
    {
            alpha[cur_n] = at;
            SV[cur_n] = xt;
            flag = true;
            cur_n ++;
    }else{
        // replace the first SV in the buffer by fpidx and fnidx
        ac = alpha[fidx];
        svc = SV[fidx];
        alpha[fidx] = at;
        SV[fidx] = xt;
        flag = true;
        ridx = fidx;
        fidx = (fidx+1) % max_n;

        // find the most similar SV for compensation
        int cidx = 0;
        double ma = 0;
        for(int i=0;i<max_n;i++){
            double temp = model.kernel_func(svc,SV[i]);
            if(temp>ma){
                cidx = i;
                ma = temp;
            }
        }
//        cout<<"The maximun sim: "<<ma<<" , idx: "<<cidx<<endl;
        alpha[cidx] += ac;
    }
}

/**
 * @brief KOIL: update threshold of decision function
 *
 * @param model the current decision function
 */
void KOIL::update_b(svm_model& model)
{
    if(model.pos_n==0||model.neg_n==0){
        model.b=0;
        return;
    }
    model.l = model.pos_n+model.neg_n;
    double* f_pos = model.predict_list(model.pos_SV,model.pos_n);
    double* f_neg = model.predict_list(model.neg_SV,model.neg_n);

    // find the min of positive value, max of negative value
    double pmin=f_pos[0], nmax=f_neg[0];
    for(int i=0;i<model.pos_n;i++){
        if(pmin>f_pos[i])
            pmin = f_pos[i];
    }

    for(int i=0;i<model.neg_n;i++){
        if(nmax<f_neg[i])
            nmax = f_neg[i];
    }

    // find the threshold
    int berr;
    double current_b = model.b;
    if(pmin>=nmax){
        model.b = current_b + (pmin+nmax)/2;
        berr = 0;
    }else{
        double bstep = (nmax-pmin)/500;
        berr = model.pos_n+model.neg_n;
        for(double cb=pmin+bstep;cb<nmax;cb+=bstep){
            int err = 0;
            for(int i=0;i<model.neg_n;i++)
                if(f_neg[i]-cb>0)
                    err++;
            for(int i=0;i<model.pos_n;i++)
                if(f_pos[i]-cb<0)
                    err++;
            if(berr > err){
                model.b = current_b + cb;
                berr = err;
            }
        }
    }
}

/**
 * @brief the calculate the AUC and Accuracy between f and y
 *
 * @param f 1xn vector, the predicted label by the model
 * @param y 1xn vector, the true label
 * @param n the number of the label
 * @return AUC AUC value
 * @return Accuracy Accuracy for the correct prediction
 */
void KOIL::evaluate_AUC(double* f, double* y, int n,
                        double& AUC, double& Accuracy)
{
    int correct = 0;
    int num_pos = 0;
    int num_neg = 0;
    int c = 0;
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            if((y[i]-y[j])*(f[i]-f[j])>0)
                correct ++;
        }
        if(y[i]>0)
            num_pos++;
        else num_neg++;
        if(f[i]*y[i]>=0)
            c++;
    }
    AUC = correct*1.0/(num_neg*num_pos);
    Accuracy = c*1.0/n;
}


/****************************************
 * koil_result: save and load KOIL result
 * **************************************/
// initial koil result
void koil_result::initial_result(int n)
{
    this->runs = n;
    this->auc  = Malloc(double,runs);
    this->accuracy = Malloc(double,runs);
    this->time = Malloc(double,runs);
    this->err_cnt = Malloc(int,runs);
}

// free the memory of the result
void koil_result::free_result()
{
    free(this->auc);
    free(this->accuracy);
    free(this->time);
    free(this->err_cnt);
}

template<class T>
// calculate the mean and std
void mean_std(T* x,int n, T& m, T& sig){
    m=0;
    for(int i=0;i<n;i++)
        m += x[i];
    m/=n;

    sig = 0;
    for(int i=0;i<n;i++){
        sig += (x[i]-m)*(x[i]-m);
    }
    sig = std::sqrt(sig/n);
}

// save koil result
void koil_result::save_result(string path, string method)
{
    cout<< "save_result path :"<<path<<endl;
    cout<<" method :"<<method<<endl;
    double mean_auc, mean_accuracy, mean_time, std_auc, std_accuracy, std_time;
    int mean_err_cnt, std_err_cnt;
    mean_std(this->auc,this->runs,mean_auc,std_auc);
    mean_std(this->accuracy,this->runs,mean_accuracy,std_accuracy);
    mean_std(this->time,this->runs,mean_time,std_time);
    mean_std(this->err_cnt,this->runs,mean_err_cnt,std_err_cnt);
    std::ofstream ofresult;
    ofresult.open(path,ios::app);
    for(int i=0;i<this->runs;i++)
        ofresult<<this->auc[i]<<"\t";
    ofresult<<endl;
    for(int i=0;i<this->runs;i++)
        ofresult<<this->accuracy[i]<<"\t";
    ofresult<<endl;
    for(int i=0;i<this->runs;i++)
        ofresult<<this->time[i]<<"\t";
    ofresult<<endl;
    for(int i=0;i<this->runs;i++)
        ofresult<<this->err_cnt[i]<<"\t";
    ofresult<<endl;

    ofresult<<"------------------------------------------------------------------------------------"<<endl;
    ofresult<<"method\t &\t AUC\t &Accuracy\t &Time\t &error counts \\\\ \n";
    ofresult<<method<<"\t &\t "<<mean_auc<<"$\\pm$"<<std_auc<<"\t &\t "
           <<mean_accuracy<<"$\\pm$"<<std_accuracy
           <<mean_time<<"$\\pm$"<<std_time
           <<mean_err_cnt<<"$\\pm$"<<std_err_cnt<<endl;
    ofresult<<"------------------------------------------------------------------------------------";
    ofresult.close();
}

// load koil result (unfinished)
void koil_result::load_result(string path){}



