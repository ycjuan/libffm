/*
The following table is the meaning of some variables in this code:

W: The pointer to the beginning of the model
w: Dynamic pointer to access values in the model
m: Number of fields
k: Number of latent factors
n: Number of features
l: Number of data points
f: Field index (0 to m-1)
d: Latent factor index (0 to k-1)
j: Feature index (0 to n-1)
i: Data point index (0 to l-1)
nnz: Number of non-zero elements
X, P: Used to store the problem in a compressed sparse row (CSR) format. len(X) = nnz, len(P) = l + 1
Y: The label. len(Y) = l
R: Precomputed scaling factor to make the 2-norm of each instance to be 1. len(R) = l
v: Value of each element in the problem
*/

#pragma GCC diagnostic ignored "-Wunused-result" 
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <new>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <cassert>
#include <numeric>

#if defined USESSE
#include <pmmintrin.h>
#endif

#if defined USEOMP
#include <omp.h>
#endif

#include "ffm.h"
#include "timer.h"

namespace ffm {

namespace {

using namespace std;

#if defined USESSE
ffm_int const kALIGNByte = 16;
#else
ffm_int const kALIGNByte = 4;
#endif

ffm_int const kALIGN = kALIGNByte/sizeof(ffm_float);
ffm_int const kCHUNK_SIZE = 10000000;
ffm_int const kMaxLineSize = 100000;

inline ffm_int get_k_aligned(ffm_int k) {
    return (ffm_int) ceil((ffm_float)k / kALIGN) * kALIGN;
}

ffm_long get_w_size(ffm_model &model) {
    ffm_int k_aligned = get_k_aligned(model.k);
    return (ffm_long) model.n * model.m * k_aligned * 2;
}

#if defined USESSE
inline ffm_float wTx(
    ffm_node *begin,
    ffm_node *end,
    ffm_float r,
    ffm_model &model, 
    ffm_float kappa=0, 
    ffm_float eta=0, 
    ffm_float lambda=0, 
    bool do_update=false) {

    ffm_int align0 = 2 * get_k_aligned(model.k);
    ffm_int align1 = model.m * align0;

    __m128 XMMkappa = _mm_set1_ps(kappa);
    __m128 XMMeta = _mm_set1_ps(eta);
    __m128 XMMlambda = _mm_set1_ps(lambda);

    __m128 XMMt = _mm_setzero_ps();

    for(ffm_node *N1 = begin; N1 != end; N1++)
    {
        ffm_int j1 = N1->j;
        ffm_int f1 = N1->f;
        ffm_float v1 = N1->v;
        if(j1 >= model.n || f1 >= model.m)
            continue;

        for(ffm_node *N2 = N1+1; N2 != end; N2++)
        {
            ffm_int j2 = N2->j;
            ffm_int f2 = N2->f;
            ffm_float v2 = N2->v;
            if(j2 >= model.n || f2 >= model.m)
                continue;

            ffm_float *w1_base = model.W + (ffm_long)j1*align1 + f2*align0;
            ffm_float *w2_base = model.W + (ffm_long)j2*align1 + f1*align0;

            __m128 XMMv = _mm_set1_ps(v1*v2*r);

            if(do_update)
            {
                __m128 XMMkappav = _mm_mul_ps(XMMkappa, XMMv);

                for(ffm_int d = 0; d < align0; d += kALIGN * 2)
                {
                    ffm_float *w1 = w1_base + d;
                    ffm_float *w2 = w2_base + d;

                    ffm_float *wg1 = w1 + kALIGN;
                    ffm_float *wg2 = w2 + kALIGN;

                    __m128 XMMw1 = _mm_load_ps(w1);
                    __m128 XMMw2 = _mm_load_ps(w2);

                    __m128 XMMwg1 = _mm_load_ps(wg1);
                    __m128 XMMwg2 = _mm_load_ps(wg2);

                    __m128 XMMg1 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw1),
                                   _mm_mul_ps(XMMkappav, XMMw2));
                    __m128 XMMg2 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw2),
                                   _mm_mul_ps(XMMkappav, XMMw1));

                    XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
                    XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));

                    XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
                    XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

                    _mm_store_ps(w1, XMMw1);
                    _mm_store_ps(w2, XMMw2);

                    _mm_store_ps(wg1, XMMwg1);
                    _mm_store_ps(wg2, XMMwg2);
                }
            }
            else
            {
                for(ffm_int d = 0; d < align0; d += kALIGN * 2)
                {
                    __m128  XMMw1 = _mm_load_ps(w1_base+d);
                    __m128  XMMw2 = _mm_load_ps(w2_base+d);

                    XMMt = _mm_add_ps(XMMt, 
                           _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
                }
            }
        }
    }

    if(do_update)
        return 0;

    XMMt = _mm_hadd_ps(XMMt, XMMt);
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    ffm_float t;
    _mm_store_ss(&t, XMMt);

    return t;
}

#else

inline ffm_float wTx(
    ffm_node *begin,
    ffm_node *end,
    ffm_float r,
    ffm_model &model, 
    ffm_float kappa=0, 
    ffm_float eta=0, 
    ffm_float lambda=0, 
    bool do_update=false) {

    ffm_int align0 = 2 * get_k_aligned(model.k);
    ffm_int align1 = model.m * align0;

    ffm_float t = 0;
    for(ffm_node *N1 = begin; N1 != end; N1++) {
        ffm_int j1 = N1->j;
        ffm_int f1 = N1->f;
        ffm_float v1 = N1->v;
        if(j1 >= model.n || f1 >= model.m)
            continue;

        for(ffm_node *N2 = N1+1; N2 != end; N2++) {
            ffm_int j2 = N2->j;
            ffm_int f2 = N2->f;
            ffm_float v2 = N2->v;
            if(j2 >= model.n || f2 >= model.m)
                continue;

            ffm_float *w1 = model.W + (ffm_long)j1*align1 + f2*align0;
            ffm_float *w2 = model.W + (ffm_long)j2*align1 + f1*align0;

            ffm_float v = v1 * v2 * r;

            if(do_update) {
                ffm_float *wg1 = w1 + kALIGN;
                ffm_float *wg2 = w2 + kALIGN;
                for(ffm_int d = 0; d < align0; d += kALIGN * 2)
                {
                    ffm_float g1 = lambda * w1[d] + kappa * w2[d] * v;
                    ffm_float g2 = lambda * w2[d] + kappa * w1[d] * v;

                    wg1[d] += g1 * g1;
                    wg2[d] += g2 * g2;

                    w1[d] -= eta / sqrt(wg1[d]) * g1;
                    w2[d] -= eta / sqrt(wg2[d]) * g2;
                }
            } else {
                for(ffm_int d = 0; d < align0; d += kALIGN * 2)
                    t += w1[d] * w2[d] * v;
            }
        }
    }

    return t;
}
#endif

ffm_float* malloc_aligned_float(ffm_long size)
{
    void *ptr;

#ifndef USESSE

    ptr = malloc(size * sizeof(ffm_float));

#else

    #ifdef _WIN32
        ptr = _aligned_malloc(size*sizeof(ffm_float), kALIGNByte);
        if(ptr == nullptr)
            throw bad_alloc();
    #else
        int status = posix_memalign(&ptr, kALIGNByte, size*sizeof(ffm_float));
        if(status != 0)
            throw bad_alloc();
    #endif

#endif
    
    return (ffm_float*)ptr;
}

ffm_model init_model(ffm_int n, ffm_int m, ffm_parameter param)
{
    ffm_model model;
    model.n = n;
    model.k = param.k;
    model.m = m;
    model.W = nullptr;
    model.normalization = param.normalization;

    ffm_int k_aligned = get_k_aligned(model.k);
    
    model.W = malloc_aligned_float((ffm_long)n*m*k_aligned*2);

    ffm_float coef = 1.0f / sqrt(model.k);
    ffm_float *w = model.W;

    default_random_engine generator;
    uniform_real_distribution<ffm_float> distribution(0.0, 1.0);

    for(ffm_int j = 0; j < model.n; j++) {
        for(ffm_int f = 0; f < model.m; f++) {
            for(ffm_int d = 0; d < k_aligned;) {
                for(ffm_int s = 0; s < kALIGN; s++, w++, d++) {
                    w[0] = (d < model.k)? coef * distribution(generator) : 0.0;
                    w[kALIGN] = 1;
                }
                w += kALIGN;
            }
        }
    }

    return model;
}

struct disk_problem_meta {
    ffm_int n = 0;
    ffm_int m = 0;
    ffm_int l = 0;
    ffm_int num_blocks = 0;
    ffm_long B_pos = 0;
    uint64_t hash1;
    uint64_t hash2;
};

struct problem_on_disk {
    disk_problem_meta meta;
    vector<ffm_float> Y;
    vector<ffm_float> R;
    vector<ffm_long> P;
    vector<ffm_node> X;
    vector<ffm_long> B;

    problem_on_disk(string path) {
        f.open(path, ios::in | ios::binary);
        if(f.good()) {
            f.read(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));
            f.seekg(meta.B_pos);
            B.resize(meta.num_blocks);
            f.read(reinterpret_cast<char*>(B.data()), sizeof(ffm_long) * meta.num_blocks);
        }
    }

    int load_block(int block_index) {
        if(block_index >= meta.num_blocks)
            assert(false);

        f.seekg(B[block_index]);

        ffm_int l;
        f.read(reinterpret_cast<char*>(&l), sizeof(ffm_int));

        Y.resize(l);
        f.read(reinterpret_cast<char*>(Y.data()), sizeof(ffm_float) * l);

        R.resize(l);
        f.read(reinterpret_cast<char*>(R.data()), sizeof(ffm_float) * l);

        P.resize(l+1);
        f.read(reinterpret_cast<char*>(P.data()), sizeof(ffm_long) * (l+1));

        X.resize(P[l]);
        f.read(reinterpret_cast<char*>(X.data()), sizeof(ffm_node) * P[l]);

        return l;
    }

    bool is_empty() {
        return meta.l == 0;
    }

private:
    ifstream f;
};

uint64_t hashfile(string txt_path, bool one_block=false)
{
    ifstream f(txt_path, ios::ate | ios::binary);
    if(f.bad())
        return 0;

    ffm_long end = (ffm_long) f.tellg();
    f.seekg(0, ios::beg);
    assert(static_cast<int>(f.tellg()) == 0);

    uint64_t magic = 90359;
    for(ffm_long pos = 0; pos < end; ) {
        ffm_long next_pos = min(pos + kCHUNK_SIZE, end);
        ffm_long size = next_pos - pos;
        vector<char> buffer(kCHUNK_SIZE);
        f.read(buffer.data(), size);

        ffm_int i = 0;
        while(i < size - 8) {
            uint64_t x = *reinterpret_cast<uint64_t*>(buffer.data() + i);
            magic = ( (magic + x) * (magic + x + 1) >> 1) + x;
            i += 8;
        }
        for(; i < size; i++) {
            char x = buffer[i];
            magic = ( (magic + x) * (magic + x + 1) >> 1) + x;
        }

        pos = next_pos;
        if(one_block)
            break;
    }

    return magic;
}

void txt2bin(string txt_path, string bin_path) {
    
    FILE *f_txt = fopen(txt_path.c_str(), "r");
    if(f_txt == nullptr)
        throw;

    ofstream f_bin(bin_path, ios::out | ios::binary);

    vector<char> line(kMaxLineSize);

    ffm_long p = 0;
    disk_problem_meta meta;

    vector<ffm_float> Y;
    vector<ffm_float> R;
    vector<ffm_long> P(1, 0);
    vector<ffm_node> X;
    vector<ffm_long> B;

    auto write_chunk = [&] () {
        B.push_back(f_bin.tellp());
        ffm_int l = Y.size();
        ffm_long nnz = P[l];
        meta.l += l;

        f_bin.write(reinterpret_cast<char*>(&l), sizeof(ffm_int));
        f_bin.write(reinterpret_cast<char*>(Y.data()), sizeof(ffm_float) * l);
        f_bin.write(reinterpret_cast<char*>(R.data()), sizeof(ffm_float) * l);
        f_bin.write(reinterpret_cast<char*>(P.data()), sizeof(ffm_long) * (l+1));
        f_bin.write(reinterpret_cast<char*>(X.data()), sizeof(ffm_node) * nnz);

        Y.clear();
        R.clear();
        P.assign(1, 0);
        X.clear();
        p = 0;
        meta.num_blocks++;
    };

    f_bin.write(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));

    while(fgets(line.data(), kMaxLineSize, f_txt)) {
        char *y_char = strtok(line.data(), " \t");

        ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;

        ffm_float scale = 0;
        for(; ; p++) {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_node N;
            N.f = atoi(field_char);
            N.j = atoi(idx_char);
            N.v = atof(value_char);

            X.push_back(N);

            meta.m = max(meta.m, N.f+1);
            meta.n = max(meta.n, N.j+1);

            scale += N.v*N.v;
        }
        scale = 1.0 / scale;

        Y.push_back(y);
        R.push_back(scale);
        P.push_back(p);

        if(X.size() > (size_t)kCHUNK_SIZE)
            write_chunk(); 
    }
    write_chunk(); 
    write_chunk(); // write a dummy empty chunk in order to know where the EOF is
    assert(meta.num_blocks == (ffm_int)B.size());
    meta.B_pos = f_bin.tellp();
    f_bin.write(reinterpret_cast<char*>(B.data()), sizeof(ffm_long) * B.size());

    fclose(f_txt);
    meta.hash1 = hashfile(txt_path, true);
    meta.hash2 = hashfile(txt_path, false);

    f_bin.seekp(0, ios::beg);
    f_bin.write(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));
}

bool check_same_txt_bin(string txt_path, string bin_path) {
    ifstream f_bin(bin_path, ios::binary | ios::ate);
    if(f_bin.tellg() < (ffm_long)sizeof(disk_problem_meta))
        return false;
    disk_problem_meta meta;
    f_bin.seekg(0, ios::beg);
    f_bin.read(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));
    if(meta.hash1 != hashfile(txt_path, true))
        return false;
    if(meta.hash2 != hashfile(txt_path, false))
        return false;

    return true;
}

} // unnamed namespace

ffm_model::~ffm_model() {
    if(W != nullptr) {
#ifndef USESSE
        free(W);
#else
    #ifdef _WIN32
        _aligned_free(W);
    #else
        free(W);
    #endif
#endif
        W = nullptr;
    }
}

void ffm_read_problem_to_disk(string txt_path, string bin_path) {

    Timer timer;
    
    cout << "First check if the text file has already been converted to binary format " << flush;
    bool same_file = check_same_txt_bin(txt_path, bin_path);
    cout << "(" << fixed << setprecision(1) << timer.toc() << " seconds)" << endl;

    if(same_file) {
        cout << "Binary file found. Skip converting text to binary" << endl;
    } else {
        cout << "Binary file NOT found. Convert text file to binary file " << flush;
        txt2bin(txt_path, bin_path);
        cout << "(" << fixed << setprecision(1) << timer.toc() << " seconds)" << endl;
    }
}

ffm_model ffm_train_on_disk(string tr_path, string va_path, ffm_parameter param) {

    problem_on_disk tr(tr_path);
    problem_on_disk va(va_path);

    ffm_model model = init_model(tr.meta.n, tr.meta.m, param);

    bool auto_stop = param.auto_stop && !va_path.empty();

    ffm_long w_size = get_w_size(model);
    vector<ffm_float> prev_W(w_size, 0);
    if(auto_stop)
        prev_W.assign(w_size, 0);
    ffm_double best_va_loss = numeric_limits<ffm_double>::max();

    cout.width(4);
    cout << "iter";
    cout.width(13);
    cout << "tr_logloss";
    if(!va_path.empty())
    {
        cout.width(13);
        cout << "va_logloss";
    }
    cout.width(13);
    cout << "tr_time";
    cout << endl;

    Timer timer;

    auto one_epoch = [&] (problem_on_disk &prob, bool do_update) {

        ffm_double loss = 0;

        vector<ffm_int> outer_order(prob.meta.num_blocks);
        iota(outer_order.begin(), outer_order.end(), 0);
        random_shuffle(outer_order.begin(), outer_order.end());
        for(auto blk : outer_order) {
            ffm_int l = prob.load_block(blk);

            vector<ffm_int> inner_order(l);
            iota(inner_order.begin(), inner_order.end(), 0);
            random_shuffle(inner_order.begin(), inner_order.end());

#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+: loss)
#endif
            for(ffm_int ii = 0; ii < l; ii++) {
                ffm_int i = inner_order[ii];

                ffm_float y = prob.Y[i];
                
                ffm_node *begin = &prob.X[prob.P[i]];

                ffm_node *end = &prob.X[prob.P[i+1]];

                ffm_float r = param.normalization? prob.R[i] : 1;

                ffm_double t = wTx(begin, end, r, model);

                ffm_double expnyt = exp(-y*t);

                loss += log1p(expnyt);

                if(do_update) {
                   
                    ffm_float kappa = -y*expnyt/(1+expnyt);

                    wTx(begin, end, r, model, kappa, param.eta, param.lambda, true);
                }
            }
        }

        return loss / prob.meta.l;
    };

    for(ffm_int iter = 1; iter <= param.nr_iters; iter++) {
        timer.tic();
        ffm_double tr_loss = one_epoch(tr, true);
        timer.toc();

        cout.width(4);
        cout << iter;
        cout.width(13);
        cout << fixed << setprecision(5) << tr_loss;

        if(!va.is_empty()) {
            ffm_double va_loss = one_epoch(va, false);

            cout.width(13);
            cout << fixed << setprecision(5) << va_loss;

            if(auto_stop) {
                if(va_loss > best_va_loss) {
                    memcpy(model.W, prev_W.data(), w_size*sizeof(ffm_float));
                    cout << endl << "Auto-stop. Use model at " << iter-1 << "th iteration." << endl;
                    break;
                } else {
                    memcpy(prev_W.data(), model.W, w_size*sizeof(ffm_float));
                    best_va_loss = va_loss; 
                }
            }
        }
        cout.width(13);
        cout << fixed << setprecision(1) << timer.get() << endl;
    }

    return model;
}

void ffm_save_model(ffm_model &model, string path) {
    ofstream f_out(path, ios::out | ios::binary);
    f_out.write(reinterpret_cast<char*>(&model.n), sizeof(ffm_int));
    f_out.write(reinterpret_cast<char*>(&model.m), sizeof(ffm_int));
    f_out.write(reinterpret_cast<char*>(&model.k), sizeof(ffm_int));
    f_out.write(reinterpret_cast<char*>(&model.normalization), sizeof(bool));

    ffm_long w_size = get_w_size(model);
    // f_out.write(reinterpret_cast<char*>(model.W), sizeof(ffm_float) * w_size);
    // Need to write chunk by chunk because some compiler use int32 and will overflow when w_size * 4 > MAX_INT

    for(ffm_long offset = 0; offset < w_size; ) {
        ffm_long next_offset = min(w_size, offset + (ffm_long) sizeof(ffm_float) * kCHUNK_SIZE);
        ffm_long size = next_offset - offset;
        f_out.write(reinterpret_cast<char*>(model.W+offset), sizeof(ffm_float) * size);
        offset = next_offset;
    }
}

ffm_model ffm_load_model(string path) {
    ifstream f_in(path, ios::in | ios::binary);

    ffm_model model;
    f_in.read(reinterpret_cast<char*>(&model.n), sizeof(ffm_int));
    f_in.read(reinterpret_cast<char*>(&model.m), sizeof(ffm_int));
    f_in.read(reinterpret_cast<char*>(&model.k), sizeof(ffm_int));
    f_in.read(reinterpret_cast<char*>(&model.normalization), sizeof(bool));

    ffm_long w_size = get_w_size(model);
    model.W = malloc_aligned_float(w_size);
    // f_in.read(reinterpret_cast<char*>(model.W), sizeof(ffm_float) * w_size);
    // Need to write chunk by chunk because some compiler use int32 and will overflow when w_size * 4 > MAX_INT

    for(ffm_long offset = 0; offset < w_size; ) {
        ffm_long next_offset = min(w_size, offset + (ffm_long) sizeof(ffm_float) * kCHUNK_SIZE);
        ffm_long size = next_offset - offset;
        f_in.read(reinterpret_cast<char*>(model.W+offset), sizeof(ffm_float) * size);
        offset = next_offset;
    }

    return model;
}

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model &model) {
    ffm_float r = 1;
    if(model.normalization) {
        r = 0;
        for(ffm_node *N = begin; N != end; N++)
            r += N->v*N->v; 
        r = 1/r;
    }

    ffm_float t = wTx(begin, end, r, model);

    return 1/(1+exp(-t));
}

} // namespace ffm
