def bootstrap(score_objs, n_boot=1000):
    """
    Given a set of DistributedROC or DistributedReliability objects, this function performs a
    bootstrap resampling of the objects and returns n_boot aggregations of them.

    Args:
        score_objs: A list of DistributedROC or DistributedReliability objects. Objects must have an __add__ method
        n_boot (int): Number of bootstrap samples

    Returns:
        An array of DistributedROC or DistributedReliability
    """
    all_samples = np.random.choice(score_objs, size=(n_boot, len(score_objs)), replace=True)
    return all_samples.sum(axis=1)