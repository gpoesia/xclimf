/*
   The MIT License (MIT)

   Copyright (c) 2014 Gabriel Poesia <gabriel.poesia at gmail.com>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "xclimf.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using std::default_random_engine;
using std::exp;
using std::fill;
using std::log;
using std::make_pair;
using std::max;
using std::min;
using std::numeric_limits;
using std::pair;
using std::pow;
using std::random_device;
using std::uniform_real_distribution;
using std::vector;

namespace {

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_prime(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// Evaluate xCLiMF's loss function (an approximation of the ERR)
double compute_loss(
        const vector<vector<double> > &users_features,
        const vector<vector<double> > &items_features,
        const vector<vector<pair<int, double>>> &ratings_matrix,
        double lambda,
        unsigned int latent_features,
        const std::function<double(double)> &relevance_probability) {
    int number_of_users = users_features.size();
    int number_of_items = items_features.size();
    double loss = 0;

    for (unsigned i = 0; i < users_features.size(); i++) {
        double user_err = 0;

        for (unsigned j = 0; j < ratings_matrix[i].size(); j++) {
            double user_j_err = log(sigmoid(xclimf::predict_score(
                            users_features[i],
                            items_features[ratings_matrix[i][j].first])));

            for (unsigned k = 0; k < ratings_matrix[i].size(); k++) {
                user_j_err +=
                    log(1 - relevance_probability(ratings_matrix[i][k].second) *
                    sigmoid(xclimf::predict_score(users_features[i],
                                items_features[ratings_matrix[i][k].first]) -
                            xclimf::predict_score(users_features[i],
                                items_features[ratings_matrix[i][j].first])));
            }

            user_err += relevance_probability(ratings_matrix[i][j].second)
                * user_j_err;
        }

        loss += user_err;
    }

    double f_user_norm = 0;
    double f_item_norm = 0;

    for (int i = 0; i < number_of_users; i++)
        for (unsigned j = 0; j < latent_features; j++)
            f_user_norm += pow(users_features[i][j], 2);

    for (int i = 0; i < number_of_items; i++)
        for (unsigned j = 0; j < latent_features; j++)
            f_item_norm += pow(items_features[i][j], 2);

    return loss - lambda / 2 * (sqrt(f_user_norm) + sqrt(f_item_norm));
}

// Gradient of the loss function w.r.t one user's latent factors
void compute_gradient_um(
        const vector<vector<double>> &users_features,
        const vector<vector<double>> &items_features,
        vector<vector<double>> &users_features_prime,
        int user_id,
        const vector<vector<pair<int, double>>> &ratings_matrix,
        unsigned int latent_features,
        double lambda,
        const std::function<double(double)> &relevance_probability) {
    fill(users_features_prime[user_id].begin(),
            users_features_prime[user_id].end(), 0.);

    for (unsigned i = 0; i < ratings_matrix[user_id].size(); i++) {
        int i_id = ratings_matrix[user_id][i].first;
        double predicted_i = xclimf::predict_score(users_features[user_id],
                items_features[i_id]);
        double relevant_p_i = relevance_probability(
                ratings_matrix[user_id][i].second);

        for (unsigned j = 0; j < latent_features; j++)
            users_features_prime[user_id][j] += relevant_p_i *
                sigmoid(-predicted_i) * items_features[i_id][j];

        for (unsigned k = 0; k < ratings_matrix[user_id].size(); k++) {
            int k_id = ratings_matrix[user_id][k].first;
            double predicted_k = xclimf::predict_score(users_features[user_id],
                    items_features[k_id]);
            double relevant_p_k = relevance_probability(
                    ratings_matrix[user_id][k].second);

            for (unsigned j = 0; j < latent_features; j++) {
                users_features_prime[user_id][j] += relevant_p_i *
                    (items_features[i_id][j] - items_features[k_id][j]) *
                    (relevant_p_k * sigmoid_prime(predicted_k - predicted_i)) /
                    (1 - relevant_p_k * sigmoid(predicted_k - predicted_i));
            }
        }
    }

    for (unsigned j = 0; j < users_features_prime[user_id].size(); j++)
        users_features_prime[user_id][j] -= lambda * users_features[user_id][j];
}

// Gradient of the loss function w.r.t one user's latent factors
void compute_gradient_vj(
        const vector<vector<double>> &users_features,
        const vector<vector<double>> &items_features,
        vector<vector<double>> &items_features_prime,
        int user_id,
        int item_index,
        const vector<vector<pair<int, double>>> &ratings_matrix,
        double lambda,
        const std::function<double(double)> &relevance_probability) {
    int item_id = ratings_matrix[user_id][item_index].first;
    fill(items_features_prime[item_id].begin(),
            items_features_prime[item_id].end(), 0.0);

    double relevant_p = relevance_probability(
            ratings_matrix[user_id][item_index].second);
    double score = xclimf::predict_score(users_features[user_id],
            items_features[item_id]);

    for (unsigned j = 0; j < items_features_prime[item_id].size(); j++) {
        items_features_prime[item_id][j] +=
            relevant_p * sigmoid(-score) * users_features[user_id][j];
    }

    for (unsigned k = 0; k < ratings_matrix[user_id].size(); k++) {
        int k_id = ratings_matrix[user_id][k].first;
        double relevant_p_k =
            relevance_probability(ratings_matrix[user_id][k].second);
        double score_k =
            xclimf::predict_score(users_features[user_id],
                    items_features[k_id]);

        for (unsigned j = 0; j < items_features_prime[item_id].size(); j++) {
            items_features_prime[item_id][j] += relevant_p * relevant_p_k *
                sigmoid_prime(score - score_k) *
                (1 / (1 - relevant_p_k * sigmoid(score_k - score)) -
                1 / (1 - relevant_p * sigmoid(score - score_k))) *
                users_features[user_id][j];
        }
    }

    for (unsigned j = 0; j < items_features_prime[item_id][j]; j++)
        items_features_prime[item_id][j] -= lambda * items_features[item_id][j];
}

void randomly_initialize(vector<vector<double>> &features) {
    random_device rd;
    default_random_engine re(rd());

    // Best range found in experiments with the MovieLens datasets
    uniform_real_distribution<> dis(0, 0.01);

    for (auto &v : features) {
        for (auto &f : v) {
            f = dis(re);
        }
    }
}

}  // namespace

namespace xclimf {

double predict_score(const vector<double> &user_features,
        const vector<double> &item_features) {
    double rating = 0;

    for (unsigned i = 0; i < user_features.size(); i++)
        rating += user_features[i] * item_features[i];

    return rating;
}

void train(
        const vector<vector<pair<int, double>>> &ratings_matrix,
        vector<vector<double>> &users_features,
        vector<vector<double>> &items_features,
        unsigned int latent_features,
        double learning_rate,
        double lambda,
        double eps,
        unsigned int max_iterations,
        bool initialize,
        const std::function<double(double)> &relevance_probability) {
    int number_of_items = 0;
    int number_of_users = ratings_matrix.size();

    for (unsigned i = 0; i < ratings_matrix.size(); i++)
        for (unsigned j = 0; j < ratings_matrix[i].size(); j++)
            number_of_items = max(number_of_items,
                    ratings_matrix[i][j].first + 1);

    users_features.resize(number_of_users);
    items_features.resize(number_of_items);

    for (int i = 0; i < number_of_items; i++)
        items_features[i].resize(latent_features);

    for (int i = 0; i < number_of_users; i++)
        users_features[i].resize(latent_features);

    if (initialize) {
        randomly_initialize(users_features);
        randomly_initialize(items_features);
    }

    vector<vector<double>> users_features_prime(number_of_users,
            vector<double>(latent_features));
    vector<vector<double>> items_features_prime(number_of_items,
            vector<double>(latent_features));

    double last_loss = -numeric_limits<double>::infinity();

    // Stochastic Gradient Descent iterations
    for (unsigned it = 0; max_iterations == 0 || it < max_iterations; it++) {
        double loss = compute_loss(users_features, items_features,
                ratings_matrix, lambda, latent_features, relevance_probability);

        if (loss < last_loss + eps)
            break;

        last_loss = loss;

        for (int i = 0; i < number_of_users; i++) {
            compute_gradient_um(users_features, items_features,
                    users_features_prime, i, ratings_matrix,
                    latent_features, lambda, relevance_probability);

            for (unsigned j = 0; j < latent_features; j++) {
                users_features[i][j] += learning_rate *
                    users_features_prime[i][j];
            }

            for (unsigned j = 0; j < ratings_matrix[i].size(); j++) {
                compute_gradient_vj(users_features, items_features,
                        items_features_prime, i, j, ratings_matrix,
                        lambda, relevance_probability);

                for (unsigned k = 0; k < latent_features; k++) {
                    items_features[ratings_matrix[i][j].first][k] +=
                        learning_rate *
                        items_features_prime[ratings_matrix[i][j].first][k];
                }
            }
        }
    }
}

// Relevance probability function used in the xCLiMF paper
// Assumes the maximum rating is 5.
double default_relevance_probability_function(double rating) {
    return (pow(2, min(rating, 5.0)) - 1) / 32.0;
}

}  // namespace xclimf
