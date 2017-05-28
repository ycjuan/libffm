#ifndef _LIBFFM_H
#define _LIBFFM_H

#include <string>

namespace ffm {

using namespace std;

typedef float ffm_float;
typedef double ffm_double;
typedef int ffm_int;
typedef long long ffm_long;

struct ffm_node {
    ffm_int f; // field index
    ffm_int j; // feature index
    ffm_float v; // value
};

struct ffm_model {
    ffm_int n; // number of features
    ffm_int m; // number of fields
    ffm_int k; // number of latent factors
    ffm_float *W = nullptr;
    bool normalization;
    ~ffm_model();
};

struct ffm_parameter {
    ffm_float eta = 0.2; // learning rate
    ffm_float lambda = 0.00002; // regularization parameter
    ffm_int nr_iters = 15;
    ffm_int k = 4; // number of latent factors
    bool normalization = true;
    bool auto_stop = false;
};

void ffm_read_problem_to_disk(string txt_path, string bin_path);

void ffm_save_model(ffm_model &model, string path);

ffm_model ffm_load_model(string path);

ffm_model ffm_train_on_disk(string Tr_path, string Va_path, ffm_parameter param);

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model &model);

} // namespace ffm

#endif // _LIBFFM_H
