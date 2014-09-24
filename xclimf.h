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

#ifndef XCLIMF_H_
#define XCLIMF_H_

#include <functional>
#include <vector>
#include <utility>

namespace xclimf {

/// Relevance probability function that is used in the xCLiMF paper.
double default_relevance_probability_function(double rating);

/// Find feature vectors for users and items using the xCLiMF algorithm.
/// \param ratings A sparse matrix of known ratings. For each user, it
/// is expected to contain a vector of pairs (i, r) where i is an integer
/// identifier of an item, and r is the rating the user assigned to that item.
/// \param users_features An output vector containing, for each user, his/her
/// extracted latent feature vector.
/// \param items_features An output vector containing, for each item, its
/// extracted latent feature vector.
/// \param d Number of latent features to extract for each user and item.
/// \param learning_rate Learning rate coefficient to use in Gradient Descent
/// \param lambda Coefficient used for regularization of the output.
/// \param eps Value to establish the stop criteria in the optimization. The
/// optimization continues until the improvement observed in the loss function
/// is less than this value.
/// \param max_iterations Maximum number of iterations of Stochastic Gradient
/// Descent. A value of 0 means there is no limit set.
/// \param initialize Whether this function should (randomly) initialize
/// the latent factors (true) or use the given initial values instead (false).
/// \param relevance_probability A function that maps a rating to a probability
/// that an item that was assigned that rating is useful to the user.
void train(
        const std::vector<std::vector<std::pair<int, double> > > &ratings,
        std::vector<std::vector<double> > &users_features,
        std::vector<std::vector<double> > &items_features,
        unsigned int d = 5,
        double learning_rate = 0.01,
        double lambda = 0.01,
        double eps = 0.01,
        unsigned int max_iterations = 20,
        bool initialize = true,
        const std::function<double(double)> &relevance_probability =
            default_relevance_probability_function);

/// Returns the predicted score of an item for a given user
/// \param user_features Latent features found by xCLiMF for the user
/// \param item_features Latent features found by xCLiMF for the item
/// user_features and item_features should have the same size. If one has
/// more elements than the other, the smaller vector is assumed to have zeros
/// in the remaining positions.
/// \return The calculated score. This score can be used for ranking the items
/// for a given user, but otherwise has no intrinsic meaning.
double predict_score(
        const std::vector<double> &user_features,
        const std::vector<double> &item_features);

};  // namespace xclimf

#endif  // XCLIMF_H_
