#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ffm.h"

using namespace std;
using namespace ffm;

string train_help()
{
    return string(
"usage: ffm-train [options] training_set_file [model_file]\n"
"\n"
"options:\n"
"-l <lambda>: set regularization parameter (default 0)\n"
"-k <factor>: set number of latent factors (default 4)\n"
"-t <iteration>: set number of iterations (default 15)\n"
"-r <eta>: set learning rate (default 0.1)\n"
"-s <nr_threads>: set number of threads (default 1)\n"
"-p <path>: set path to the validation set\n"
"--quiet: quiet model (no output)\n"
"--norm: do instance-wise normalization\n"
"--no-rand: disable random update\n");
}

struct Option
{
    Option() : param(ffm_get_default_param()), nr_folds(1), do_cv(false) {}
    string tr_path, va_path, model_path;
    ffm_parameter param;
    ffm_int nr_folds;
    bool do_cv;
};

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option opt;

    ffm_int i = 1;
    for(; i < argc; i++)
    {
        if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of iterations after -t");
            i++;
            opt.param.nr_iters = stoi(args[i]);
            if(opt.param.nr_iters <= 0)
                throw invalid_argument("number of iterations should be greater than zero");
        }
        else if(args[i].compare("-k") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of factors after -k");
            i++;
            opt.param.k = stoi(args[i]);
            if(opt.param.k <= 0)
                throw invalid_argument("number of factors should be greater than zero");
        }
        else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify eta after -r");
            i++;
            opt.param.eta = stof(args[i]);
            if(opt.param.eta <= 0)
                throw invalid_argument("learning rate should be greater than zero");
        }
        else if(args[i].compare("-l") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify lambda after -l");
            i++;
            opt.param.lambda = stof(args[i]);
            if(opt.param.lambda < 0)
                throw invalid_argument("regularization cost should not be smaller than zero");
        }
        else if(args[i].compare("-s") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of threads after -s");
            i++;
            opt.param.nr_threads = stoi(args[i]);
            if(opt.param.nr_threads <= 0)
                throw invalid_argument("number of threads should be greater than zero");
        }
        else if(args[i].compare("-v") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of folds after -v");
            i++;
            opt.nr_folds = stoi(args[i]);
            if(opt.nr_folds <= 1)
                throw invalid_argument("number of folds should be greater than one");
            opt.do_cv = true;
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;
            opt.va_path = args[i];
        }
        else if(args[i].compare("--norm") == 0)
        {
            opt.param.normalization= true;
        }
        else if(args[i].compare("--quiet") == 0)
        {
            opt.param.quiet = true;
        }
        else if(args[i].compare("--no-rand") == 0)
        {
            opt.param.random = false;
        }
        else
        {
            break;
        }
    }

    if(i != argc-2 && i != argc-1)
        throw invalid_argument("cannot parse command\n");

    opt.tr_path = args[i];
    i++;

    if(i < argc)
    {
        opt.model_path = string(args[i]);
    }
    else if(i == argc)
    {
        const char *ptr = strrchr(&*opt.tr_path.begin(), '/');
        if(!ptr)
            ptr = opt.tr_path.c_str();
        else
            ++ptr;
        opt.model_path = string(ptr) + ".model";
    }
    else
    {
        throw invalid_argument("cannot parse argument");
    }

    return opt;
}

ffm_problem read_problem(string path)
{
    int const kMaxLineSize = 1000000;

    ffm_problem prob;
    prob.l = 0;
    prob.n = 0;
    prob.m = 0;
    prob.X = nullptr;
    prob.P = nullptr;
    prob.Y = nullptr;

    if(path.empty())
        return prob;

    FILE *f = fopen(path.c_str(), "r");
    if(f == nullptr)
        throw runtime_error("cannot open " + path);

    char line[kMaxLineSize];

    ffm_long nnz = 0;
    for(ffm_int i = 0; fgets(line, kMaxLineSize, f) != nullptr; i++, prob.l++)
    {
        strtok(line, " \t");
        for(; ; nnz++)
        {
            char *field_char = strtok(nullptr,":");
            strtok(nullptr,":");
            strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;
        }
    }
    rewind(f);

    prob.X = new ffm_node[nnz];
    prob.P = new ffm_long[prob.l+1];
    prob.Y = new ffm_float[prob.l];

    ffm_long p = 0;
    prob.P[0] = 0;
    for(ffm_int i = 0; fgets(line, kMaxLineSize, f) != nullptr; i++)
    {
        char *y_char = strtok(line, " \t");
        ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;
        prob.Y[i] = y;

        for(; ; ++p)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_int field = atoi(field_char);
            ffm_int idx = atoi(idx_char);
            ffm_float value = atof(value_char);

            prob.m = max(prob.m, field+1);
            prob.n = max(prob.n, idx+1);
            
            prob.X[p].f = field;
            prob.X[p].j = idx;
            prob.X[p].v = value;
        }
        prob.P[i+1] = p;
    }

    fclose(f);

    return prob;
}

void destroy_problem(ffm_problem &prob)
{
    delete[] prob.X;
    delete[] prob.P;
    delete[] prob.Y;
}

int main(int argc, char **argv)
{
    Option opt;
    try
    {
        opt = parse_option(argc, argv);
    }
    catch(invalid_argument &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    ffm_problem tr, va;
    try
    {
        tr = read_problem(opt.tr_path);
        va = read_problem(opt.va_path);
    }
    catch(runtime_error &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    if(opt.do_cv)
    {
        ffm_cross_validation(&tr, opt.nr_folds, opt.param);
    }
    else
    {
        ffm_model *model = train_with_validation(&tr, &va, opt.param);

        ffm_int status = ffm_save_model(model, opt.model_path.c_str());
        if(status != 0)
        {
            destroy_problem(tr);
            destroy_problem(va);
            ffm_destroy_model(&model);

            return 1;
        }

        ffm_destroy_model(&model);
    }

    destroy_problem(tr);
    destroy_problem(va);

    return 0;
}
