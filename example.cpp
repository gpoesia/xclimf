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

#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

using namespace std;

#include "./xclimf.h"

vector<vector<pair<int, double>>> read_input(
        istream &input, vector<int> &ids_to_items) {
    vector<vector<pair<int, double>>> ratings;
    map<int, int> item_ids;

    string line;
    while (getline(input, line)) {
        ratings.resize(ratings.size() + 1);

        istringstream iss(line);

        string rating;

        while (iss >> rating) {
            unsigned item_id = 0, rating_value = 0;

            int i = 0;

            while (rating[i] != ':') {
                item_id = 10*item_id + (rating[i] - '0');
                i++;
            }

            rating_value = rating[i+1] - '0';

            unsigned item_index = 0;

            if (item_ids.count(item_id) == 0) {
                item_index = ids_to_items.size();
                ids_to_items.push_back(item_id);
                item_ids[item_id] = item_index;
            } else {
                item_index = item_ids[item_id];
            }

            ratings.back().push_back(make_pair(item_index, rating_value));
        }
    }

    return ratings;
}

void print_output(ostream &output,
        vector<vector<double> > &users_features,
        vector<vector<double> > &items_features,
        const vector<int> &ids_to_items) {
    for (unsigned i = 0; i < users_features.size(); i++) {
        output << ids_to_items[0] << ":"
            << xclimf::predict_score(users_features[i], items_features[0]);

        for (unsigned j = 1; j < items_features.size(); j++) {
            output << " " << ids_to_items[j] << ":"
                << xclimf::predict_score(users_features[i], items_features[j]);
        }

        output << '\n';
    }
}

int main() {
    vector<int> ids_to_items;
    vector<vector<pair<int, double>>> ratings_matrix;
    ratings_matrix = read_input(cin, ids_to_items);

    vector<vector<double>> items_features;
    vector<vector<double>> users_features;

    xclimf::train(ratings_matrix, users_features, items_features);
    print_output(cout, users_features, items_features, ids_to_items);

    return 0;
}
