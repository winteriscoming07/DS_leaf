## Coverage of authoritative leaf IDs
   role     species  images  matched_authoritative_id  pct_matched  no_sheet_upstream  not_in_sheet  ambiguous  distinct_leaves
  known       Apple   250.0                     250.0        100.0                0.0           0.0        0.0            161.0
  known        Corn   250.0                       0.0          0.0              250.0           0.0        0.0              0.0
  known       Grape   250.0                       0.0          0.0              250.0           0.0        0.0              0.0
  known       Peach   250.0                     250.0        100.0                0.0           0.0        0.0             56.0
  known      Tomato   250.0                     150.0         60.0                0.0         100.0        0.0            122.0
unknown Bell pepper   150.0                     150.0        100.0                0.0           0.0        0.0             93.0
unknown      Potato   150.0                     150.0        100.0                0.0           0.0        0.0             38.0

## Leaves crossing train/val/test (matched images only)
### primary (image-level)
species  images  leaves  leaves_crossing  images_in_crossing_leaves  test_images  test_images_sharing_leaf_with_train_or_val
  Apple     250     161               41                         97           38                                          24
  Peach     250      56               42                        210           38                                          38
 Tomato     150     122               16                         37           25                                           9
### frame-block heuristic
species  images  leaves  leaves_crossing  images_in_crossing_leaves  test_images  test_images_sharing_leaf_with_train_or_val
  Apple     243     159                1                          3           35                                           0
  Peach     242      56                0                          0           34                                           0
 Tomato     146     119                0                          0           22                                           0
### hybrid leaf-map
species  images  leaves  leaves_crossing  images_in_crossing_leaves  test_images  test_images_sharing_leaf_with_train_or_val
  Apple     250     161                0                          0           36                                           0
  Peach     250      56                0                          0           36                                           0
 Tomato     150     122                0                          0           22                                           0

## Leaf-clustered 95% bootstrap intervals (2000 resamples)
primary (image-level)    hog_svm    known_accuracy=0.973-1.000  unknown_rejection=0.088-0.176  auroc=0.729-0.820  (clusters: known 177, unknown 131)
primary (image-level)    mobilenet  known_accuracy=1.000-1.000  unknown_rejection=0.668-0.789  auroc=0.919-0.971  (clusters: known 177, unknown 131)
frame-block heuristic    hog_svm    known_accuracy=0.919-0.984  unknown_rejection=0.142-0.251  auroc=0.697-0.799  (clusters: known 143, unknown 131)
frame-block heuristic    mobilenet  known_accuracy=1.000-1.000  unknown_rejection=0.590-0.735  auroc=0.884-0.950  (clusters: known 143, unknown 131)
hybrid leaf-map          hog_svm    known_accuracy=0.926-0.984  unknown_rejection=0.083-0.169  auroc=0.752-0.844  (clusters: known 141, unknown 131)
hybrid leaf-map          mobilenet  known_accuracy=0.983-1.000  unknown_rejection=0.668-0.800  auroc=0.859-0.934  (clusters: known 141, unknown 131)
