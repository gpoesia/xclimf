# libxclimf: An implementation of xCLiMF/CLiMF
Copyright (c) 2014 Gabriel Poesia <gabriel.poesia at gmail.com>

## Description

Implementation of the xCLiMF algorithm presented in the following paper:

Shi, Yue, et al. "xCLiMF: optimizing expected reciprocal rank for data 
with multiple levels of relevance."
Proceedings of the 7th ACM conference on Recommender systems. ACM, 2013.

xCLiMF is a List-wise Learning to Rank algorithm for Recommender Systems.
It takes as input a matrix with known ratings that users gave to some items,
and produces as output two matrices, one for users and one for items.
Each of these matrices has a (latent) feature vector for each user or item.
The predicted score of an item for a user is then given by the inner product
of their feature vectors. These scores have no intrinsic meaning, and
are optimized for ranking, not rating prediction.
They only serve the purpose of sorting the items by (predicted) preference
for each user. Using this, one can produce recommendations.

xCLiMF is a generalization of CLiMF. This implementation can be easily
used to work with CLiMF without modifying the source code.
All you need to do is to provide a relevance
probability function that assigns a probability of one to items that
are relevant and a probability of zero otherwise. For example, in the
common scenario in which ratings are integers ranging from 1 to 5,
it's usual to consider items rated 4 and 5 as relevant, and the remaining
ones as irrelevant. To use CLiMF with this criterion, you'd provide
a relevance probability function that returns 1 if the rating received
is greater than or equal to 5, and 0 otherwise.

## Example 

An example program is given in `example.cpp`. It reads a matrix from 
standard input with each line containing the known ratings for one user.
The ratings are given in the sparse format ``item_id:rating``. For example,
1:2 means the user gave a rating of 2 to item 1. For example, the following
is a valid input file:

```
1:2 3:5 4:5
2:3 6:1 3:2
5:5 6:5
6:2 1:5 2:5
```

This input has 4 users. The first, second and fourth have 3 known ratings each,
and the third has two. The example program will use this file as a training
set for ListRank-MF and output another file in the same format, but with
the predicted scores for each user with respect to each item. Running the 
program given the input above may give the following output:

```
1:0.000178097 3:0.000193224 4:0.000158455 2:0.000140313 6:0.000159095 5:8.11646e-05
1:9.84883e-05 3:0.000182623 4:0.000126382 2:0.000117597 6:0.00011757 5:8.91539e-05
1:0.000142499 3:0.000219044 4:0.000161765 2:0.000127183 6:0.000124615 5:9.01645e-05
1:0.000204914 3:0.000180513 4:0.000141306 2:0.000176814 6:0.000209659 5:8.44147e-05
1:0.000121719 3:0.000195112 4:0.000145771 2:0.000142094 6:0.000148575 5:9.41174e-05
```

Using this output, we can infer user 1 would prefer to see item 2 (which
has a score of 0.00019 for him) than item 6 (which has a lower predicted score
of 0.00015). Note that the predicted scores tend to be consistent (with
respect to their relative order) with the observed ratings given in the
input.

## License

This library is released under the MIT license. In practice, this means
you can use it freely provided that you keep the copyright notice.

## Last words

If you find this library useful for any purpose, I'd be very pleased 
to hear about that! I'll appreciate if you send me an email simply
telling me what you are using libxclimf for (research, personal
projects, etc).
