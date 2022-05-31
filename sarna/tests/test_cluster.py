import numpy as np
import sarna


def test_compute_threshold_via_permutations():
    """Make sure that threshold computed through permutations is correct.

    Check that the threshold computed through permutations/randomization
    on data that fulfills assumptions of analytical tests is sufficiently
    close to the analytical threshold.
    """
    n_groups = 2

    for paired in [False, True]:
        if paired:
            n_obs = [101, 101]
            data = [np.random.randn(n_obs[0])]
            data.append(data[0] + np.random.randn(n_obs[0]))
        else:
            n_obs = [102, 100]
            data = [np.random.randn(n) for n in n_obs]

        analytical_threshold = sarna.cluster._compute_threshold(
            data=data, threshold=None, p_threshold=0.05, paired=paired,
            one_sample=False)

        stat_fun = sarna.cluster._find_stat_fun(
            n_groups, paired=paired, tail='both')

        permutation_threshold = (
            sarna.cluster._compute_threshold_via_permutations(
                data, paired=paired, tail='both', stat_fun=stat_fun,
                n_permutations=2_000, progress=False
            )
        )

        avg_perm = np.abs(permutation_threshold).mean()
        error = analytical_threshold - avg_perm

        print('paired:', paired)
        print('analytical_threshold:', analytical_threshold)
        print('permutation threshold:', permutation_threshold)
        print('average permutation threshold:', avg_perm)
        print('difference:', error)

        assert np.abs(error) < 0.15
