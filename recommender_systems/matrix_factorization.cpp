#include <iostream>
#include <unordered_map>
#include <numeric>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

vector<double *> prefs;
int n_iters = 100;

class Err
{
public:
    double train;

public:
    double test;
};

class Calculator
{
    unordered_map<long, double *> u_factors;
    unordered_map<long, double *> i_factors;

    int n_factors = 2;

    // Stochastic Gradient descent
    float alpha = 0.03;
    float my_lambda = 0.1;

    Err min_test_err = Err{
        train : 1,
        test : 1,
    };

public:
    Calculator(int factors, float alpha, float lambda)
    {
        n_factors = factors;
        alpha = alpha;
        my_lambda = lambda;
    }

public:
    void process(bool print)
    {
        std::random_device rd;
        std::default_random_engine eng(rd());

        fill_random_factors(eng, n_factors);

        std::shuffle(std::begin(prefs), std::end(prefs), eng);
        long end = get_partial_prefs(0.9);

        printf("Initial error: %f\n", calc_error(0, end));

        for (int t = 0; t < n_iters; t++)
        {
            std::shuffle(std::begin(prefs), std::begin(prefs) + end, eng);

            for (long r = 0; r < end; r++)
            {
                long u = prefs[r][0];
                long i = prefs[r][1];

                float error = prefs[r][2] - dot_product(u_factors[u], i_factors[i], n_factors);
                for (int k = 0; k < n_factors; k++)
                {
                    u_factors[u][k] = u_factors[u][k] + alpha * (error * i_factors[i][k] - my_lambda * u_factors[u][k]);
                    i_factors[i][k] = i_factors[i][k] + alpha * (error * u_factors[u][k] - my_lambda * i_factors[i][k]);
                }
            }

            Err err = Err{
                train : calc_error(0, end),
                test : calc_error(end, prefs.size()),
            };

            if (print == true)
            {
                printf("Iteration %d, Train error: %f Test error: %f\n", t, err.train, err.test);
            }

            if (err.test < min_test_err.test)
            {
                min_test_err = err;
            }
        }
    }

public:
    Err get_min_err()
    {
        return min_test_err;
    }

    double calc_error(long start, long end)
    {
        long u_idx, i_idx;
        double error = 0;
        for (long i = start; i < end; i++)
        {
            u_idx = prefs[i][0];
            i_idx = prefs[i][1];

            error += abs(prefs[i][2] - dot_product(u_factors[u_idx], i_factors[i_idx], n_factors));
        }
        return error / (end - start);
    }

    void fill_random_factors(std::default_random_engine eng, int n_factors)
    {
        double MIN = -0.5;
        double MAX = 0.5;

        std::uniform_real_distribution<double> distr(MIN, MAX);
        for (long r = 0; r < prefs.size(); r++)
        {
            double *arr = new double[n_factors];
            for (int i = 0; i < n_factors; i++)
            {
                arr[i] = distr(eng);
            }
            u_factors[prefs[r][0]] = arr;

            arr = new double[n_factors];
            for (int i = 0; i < n_factors; i++)
            {
                arr[i] = distr(eng);
            }
            i_factors[prefs[r][1]] = arr;
        }
    }

    long get_partial_prefs(double percent)
    {
        return (long)prefs.size() * percent;
    }

    double dot_product(double *v1, double *v2, int size)
    {
        double product = 0;
        for (int i = 0; i < size; i++)
        {
            product += v1[i] * v2[i];
        }
        return product;
    }
};

void read_file()
{
    int f_cols = 4;

    ifstream fin;
    fin.open("../datasets/ml-latest-small/ratings.csv");
    if (!fin.is_open())
    {
        cerr << "error: cannot open file\n";
    }

    string line, val;
    int i = 0;
    int k;
    while (getline(fin, line))
    {
        if (i == 0)
        {
            i++;
            continue;
        }
        double *f = new double[f_cols];
        stringstream s(line);
        k = 0;
        while (getline(s, val, ','))
        {
            f[k] = (stod(val));
            k++;
        }
        prefs.push_back(f);
    }

    /*
        i = 0;
        for (double* row : prefs) {
            for (k=0; k<f_cols; k++)
                cout << row[k] << "  ";
            cout << "\n";
            i++;
            if (i > 10)
                break;
        }
    */
}

/*
double calc_error(vector<vector<double> > X, unordered_map<long, vector<double> > u_factors, unordered_map<long, vector<double> > i_factors){
    long rows = X.size();
    long cols = X[0].size();
    long u_idx, i_idx;
    double error = 0;
    for (long i=0; i<rows; i++){
        u_idx = X[i][0];
        i_idx = X[i][1];

        error += abs(X[i][2] - inner_product(u_factors[u_idx].begin(), u_factors[u_idx].end(), i_factors[i_idx].begin(), 0));
    }
    return error/rows;

}
*/

int random_int(int min, int max)
{
    static bool first = true;
    if (first)
    {
        srand(time(NULL)); // seeding for the first time only!
        first = false;
    }
    return min + rand() % ((max + 1) - min);
}

vector<int> n_random_int(int min, int max, int n)
{
    vector<int> vec;
    for (size_t i = 0; i < n; i++)
    {
        int num = random_int(min, max);
        vec.push_back(num);
    }
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    return vec;
}

vector<float> n_random_float(int min, int max, int n, int divider)
{
    vector<float> vec;
    for (size_t i = 0; i < n; i++)
    {
        float num = float(random_int(min, max)) / divider;
        vec.push_back(num);
    }
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
    return vec;
}

Err grid_opt()
{
    int factors_min = 2,
        factors_max = 5;

    int alpha_min = 3,
        alpha_max = 8;

    int lambda_min = 10,
        lambda_max = 20;

    vector<int> factors = n_random_int(2, 4, 2);
    vector<float> alphas = n_random_float(3, 15, 3, 100);
    vector<float> lambdas = n_random_float(1, 7, 2, 10);
    Err min_test_err = Err{
        train : 1,
        test : 1,
    };
    for (size_t i = 0; i < factors.size(); i++)
    {
        int factor = factors[i];
        for (size_t j = 0; j < alphas.size(); j++)
        {
            float alpha = alphas[j];
            for (size_t k = 0; k < lambdas.size(); k++)
            {
                float lambda = lambdas[k];

                printf("Calculating with => factors: %d, alpha: %f, lambda: %f\n", factor, alpha, lambda);
                Calculator pass = Calculator(factor, alpha, lambda);
                pass.process(false);

                Err min_err = pass.get_min_err();
                printf("Min Test error: %f\n", min_err.test);
                printf("===========================\n");

                if (min_err.test < min_test_err.test)
                {
                    min_test_err = min_err;
                }
            }
        }
    }

    return min_test_err;
}

void sgd()
{
    Err min_grid_err = grid_opt();
    printf("Grid Min Test error: %f\n", min_grid_err.test);
}

int main()
{
    read_file();
    sgd();

    return 0;
}
