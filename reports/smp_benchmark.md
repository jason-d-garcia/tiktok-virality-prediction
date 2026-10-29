# SMP-Video benchmark

5-fold creator-disjoint CV on the labeled train split; target is the challenge label log2(views/day) + 1. Lower MAE/MSE and higher SRC are better.

| model | features | mae | mse | src | mape | src_fold_sd |
| --- | --- | --- | --- | --- | --- | --- |
| hgb | meta+text+clip+user | 1.578 | 3.856 | 0.699 | 0.259 | 0.017 |
| ridge | meta+text+clip+user | 1.626 | 4.262 | 0.686 | 0.266 | 0.018 |
| ridge | followers only | 1.933 | 5.471 | 0.535 | 0.327 | 0.023 |
| ridge | meta+text+clip | 2.102 | 6.669 | 0.434 | 0.385 | 0.034 |
| hgb | meta+text+clip | 2.121 | 6.735 | 0.420 | 0.391 | 0.028 |
| ridge | meta+text | 2.216 | 7.320 | 0.326 | 0.412 | 0.039 |
| hgb | meta+text | 2.219 | 7.292 | 0.329 | 0.409 | 0.034 |
| ridge | meta | 2.367 | 8.080 | 0.128 | 0.438 | 0.036 |
| hgb | meta | 2.369 | 8.130 | 0.109 | 0.440 | 0.040 |
| mean | - | 2.393 | 8.205 | -0.027 | 0.444 | nan |

## Conformal intervals: meta+text+clip

Coverage overall and by creator-size quartile, with mean width (label units):

| alpha | method | all | Q1 small | Q2 | Q3 | Q4 large | width | interval_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.100 | raw quantile (no conformal) | 0.859 | 0.766 | 0.920 | 0.910 | 0.840 | 8.343 | 10.543 |
| 0.100 | split | 0.906 | 0.797 | 0.945 | 0.944 | 0.939 | 8.659 | 10.536 |
| 0.100 | cqr | 0.899 | 0.823 | 0.941 | 0.940 | 0.895 | 9.103 | 10.418 |
| 0.100 | mondrian split | 0.907 | 0.907 | 0.902 | 0.911 | 0.908 | 8.574 | 10.143 |
| 0.100 | mondrian cqr | 0.902 | 0.913 | 0.897 | 0.903 | 0.897 | 9.002 | 10.189 |
| 0.200 | raw quantile (no conformal) | 0.725 | 0.610 | 0.793 | 0.784 | 0.714 | 6.340 | 9.274 |
| 0.200 | split | 0.804 | 0.679 | 0.847 | 0.860 | 0.832 | 6.801 | 9.011 |
| 0.200 | cqr | 0.795 | 0.677 | 0.854 | 0.860 | 0.791 | 7.209 | 9.120 |
| 0.200 | mondrian split | 0.803 | 0.817 | 0.792 | 0.795 | 0.808 | 6.876 | 8.824 |
| 0.200 | mondrian cqr | 0.800 | 0.822 | 0.815 | 0.775 | 0.789 | 7.265 | 8.976 |

Coverage by predicted level (tercile of the point prediction):

| alpha | method | pred high | pred low | pred mid |
| --- | --- | --- | --- | --- |
| 0.100 | raw quantile (no conformal) | 0.856 | 0.841 | 0.879 |
| 0.100 | split | 0.938 | 0.882 | 0.898 |
| 0.100 | cqr | 0.898 | 0.891 | 0.910 |
| 0.100 | mondrian split | 0.943 | 0.887 | 0.891 |
| 0.100 | mondrian cqr | 0.884 | 0.912 | 0.911 |
| 0.200 | raw quantile (no conformal) | 0.724 | 0.710 | 0.742 |
| 0.200 | split | 0.873 | 0.749 | 0.792 |
| 0.200 | cqr | 0.807 | 0.769 | 0.811 |
| 0.200 | mondrian split | 0.861 | 0.766 | 0.784 |
| 0.200 | mondrian cqr | 0.788 | 0.804 | 0.809 |

## Conformal intervals: meta+text+clip+user

Coverage overall and by creator-size quartile, with mean width (label units):

| alpha | method | all | Q1 small | Q2 | Q3 | Q4 large | width | interval_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.100 | raw quantile (no conformal) | 0.833 | 0.812 | 0.837 | 0.860 | 0.822 | 6.351 | 8.659 |
| 0.100 | split | 0.901 | 0.933 | 0.863 | 0.879 | 0.929 | 6.365 | 8.073 |
| 0.100 | cqr | 0.903 | 0.904 | 0.906 | 0.911 | 0.893 | 7.157 | 8.413 |
| 0.100 | mondrian split | 0.902 | 0.887 | 0.893 | 0.904 | 0.924 | 6.427 | 8.085 |
| 0.100 | mondrian cqr | 0.904 | 0.904 | 0.899 | 0.901 | 0.915 | 7.206 | 8.454 |
| 0.200 | raw quantile (no conformal) | 0.709 | 0.687 | 0.717 | 0.747 | 0.684 | 4.897 | 7.221 |
| 0.200 | split | 0.802 | 0.841 | 0.751 | 0.770 | 0.847 | 5.054 | 6.845 |
| 0.200 | cqr | 0.802 | 0.798 | 0.819 | 0.818 | 0.774 | 5.626 | 7.081 |
| 0.200 | mondrian split | 0.803 | 0.792 | 0.788 | 0.799 | 0.835 | 5.038 | 6.820 |
| 0.200 | mondrian cqr | 0.799 | 0.788 | 0.805 | 0.787 | 0.814 | 5.594 | 7.097 |

Coverage by predicted level (tercile of the point prediction):

| alpha | method | pred high | pred low | pred mid |
| --- | --- | --- | --- | --- |
| 0.100 | raw quantile (no conformal) | 0.830 | 0.811 | 0.856 |
| 0.100 | split | 0.945 | 0.902 | 0.854 |
| 0.100 | cqr | 0.903 | 0.885 | 0.921 |
| 0.100 | mondrian split | 0.943 | 0.881 | 0.878 |
| 0.100 | mondrian cqr | 0.913 | 0.882 | 0.917 |
| 0.200 | raw quantile (no conformal) | 0.713 | 0.686 | 0.726 |
| 0.200 | split | 0.860 | 0.791 | 0.751 |
| 0.200 | cqr | 0.811 | 0.787 | 0.809 |
| 0.200 | mondrian split | 0.873 | 0.775 | 0.758 |
| 0.200 | mondrian cqr | 0.823 | 0.769 | 0.802 |
