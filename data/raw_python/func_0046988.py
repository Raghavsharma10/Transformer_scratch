def extract(features, groups,
            weight_method=default_weight_method,
            num_bins=default_num_bins,
            edge_range=default_edge_range,
            trim_outliers=default_trim_behaviour,
            trim_percentile=default_trim_percentile,
            use_original_distribution=False,
            relative_to_all=False,
            asymmetric=False,
            return_networkx_graph=default_return_networkx_graph,
            out_weights_path=default_out_weights_path):
    """
    Extracts the histogram-distance weighted adjacency matrix.

    Parameters
    ----------
    features : ndarray or str
        1d array of scalar values, either provided directly as a 1d numpy array,
        or as a path to a file containing these values

    groups : ndarray or str
        Membership array of same length as `features`, each value specifying which group that particular node belongs to.
        Input can be either provided directly as a 1d numpy array,or as a path to a file containing these values.

        For example, if you have cortical thickness values for 1000 vertices (`features` is ndarray of length 1000),
        belonging to 100 patches, the groups array (of length 1000) could  have numbers 1 to 100 (number of unique values)
        specifying which element belongs to which cortical patch.

        Grouping with numerical values (contiguous from 1 to num_patches) is strongly recommended for simplicity,
        but this could also be a list of strings of length p, in which case a tuple is returned,
        identifying which weight belongs to which pair of patches.

    weight_method : string or callable, optional
        Type of distance (or metric) to compute between the pair of histograms.
        It can either be a string identifying one of the weights implemented below, or a valid callable.

        If a string, it must be one of the following methods:

        - 'chebyshev'
        - 'chebyshev_neg'
        - 'chi_square'
        - 'correlate'
        - 'correlate_1'
        - 'cosine'
        - 'cosine_1'
        - 'cosine_2'
        - 'cosine_alt'
        - 'euclidean'
        - 'fidelity_based'
        - 'histogram_intersection'
        - 'histogram_intersection_1'
        - 'jensen_shannon'
        - 'kullback_leibler'
        - 'manhattan'
        - 'minowski'
        - 'noelle_1'
        - 'noelle_2'
        - 'noelle_3'
        - 'noelle_4'
        - 'noelle_5'
        - 'relative_bin_deviation'
        - 'relative_deviation'

        Note only the following are *metrics*:

        - 'manhattan'
        - 'minowski'
        - 'euclidean'
        - 'noelle_2'
        - 'noelle_4'
        - 'noelle_5'

        The following are *semi- or quasi-metrics*:

        - 'kullback_leibler'
        - 'jensen_shannon'
        - 'chi_square'
        - 'chebyshev'
        - 'cosine_1'
        - 'chebyshev_neg'
        - 'correlate_1'
        - 'histogram_intersection_1'
        - 'relative_deviation'
        - 'relative_bin_deviation'
        - 'noelle_1'
        - 'noelle_3'

        The following are  classified to be similarity functions:

        - 'histogram_intersection'
        - 'correlate'
        - 'cosine'
        - 'cosine_2'
        - 'cosine_alt'
        - 'fidelity_based'

        *Default* choice: 'minowski'.

        The method can also be one of the following identifying metrics that operate on the original data directly -
         e.g. difference in the medians coming from the distributions of the pair of ROIs.

         - 'diff_medians'
         - 'diff_means'
         - 'diff_medians_abs'
         - 'diff_means_abs'

         Please note this can lead to adjacency matrices that may not be symmetric
            e.g. difference metric on two scalars is not symmetric).
            In this case, be sure to use the flag: allow_non_symmetric=True

        If weight_method is a callable, it must two accept two arrays as input and return one scalar as output.
            Example: ``diff_in_skew = lambda x, y: abs(scipy.stats.skew(x)-scipy.stats.skew(y))``
            NOTE: this method will be applied to histograms (not the original distribution of features from group/ROI).
            In order to apply this callable directly on the original distribution (without trimming and histogram binning),
            use ``use_original_distribution=True``.

    num_bins : scalar, optional
        Number of bins to use when computing histogram within each patch/group.

        Note:

        1) Please ensure same number of bins are used across different subjects
        2) histogram shape can vary widely with number of bins (esp with fewer bins in the range of 3-20), and hence the features extracted based on them vary also.
        3) It is recommended to study the impact of this parameter on the final results of the experiment.

        This could also be optimized within an inner cross-validation loop if desired.

    edge_range : tuple or None
        The range of edges within which to bin the given values.
        This can be helpful to ensure correspondence across multiple invocations of hiwenet (for different subjects),
        in terms of range across all bins as well as individual bin edges.
        Default is to automatically compute from the given values.

        Accepted format:

            - tuple of finite values: (range_min, range_max)
            - None, triggering automatic calculation (default)

        Notes : when controlling the ``edge_range``, it is not possible trim the tails (e.g. using the parameters
        ``trim_outliers`` and ``trim_percentile``) for the current set of features using its own range.

    trim_outliers : bool, optional
        Whether to trim a small percentile of outliers at the edges of feature range,
        when features are expected to contain extreme outliers (like 0 or eps or Inf).
        This is important to avoid numerical problems and also to stabilize the weight estimates.

    trim_percentile : float
        Small value specifying the percentile of outliers to trim.
        Default: 5 (5%). Must be in open interval (0, 100).

    use_original_distribution : bool, optional
        When using a user-defined callable, this flag
        1) allows skipping of pre-processing (trimming outliers) and histogram construction,
        2) enables the application of arbitrary callable (user-defined) on the original distributions coming from the two groups/ROIs/nodes directly.

        Example: ``diff_in_medians = lambda x, y: abs(np.median(x)-np.median(y))``

        This option is valid only when weight_method is a valid callable,
            which must take two inputs (possibly of different lengths) and return a single scalar.

    relative_to_all : bool
        Flag to instruct the computation of a grand histogram (distribution pooled from values in all ROIs),
        and compute distances (based on distance specified by ``weight_method``) by from each ROI to the grand mean.
        This would result in only N distances for N ROIs, instead of the usual N*(N-1) pair-wise distances.

    asymmetric : bool
        Flag to identify resulting adjacency matrix is expected to be non-symmetric.
        Note: this results in twice the computation time!
        Default: False , for histogram metrics implemented here are symmetric.

    return_networkx_graph : bool, optional
        Specifies the need for a networkx graph populated with weights computed. Default: False.

    out_weights_path : str, optional
        Where to save the extracted weight matrix. If networkx output is returned, it would be saved in GraphML format.
        Default: nothing saved unless instructed.

    Returns
    -------
    edge_weights : ndarray
        numpy 2d array of pair-wise edge-weights (of size: num_groups x num_groups),
        wherein num_groups is determined by the total number of unique values in `groups`.

        **Note**:

        - Only the upper triangular matrix is filled as the distance between node i and j would be the same as j and i.
        - The edge weights from the upper triangular matrix can easily be obtained by

        .. code-block:: python

            weights_array = edge_weights[ np.triu_indices_from(edge_weights, 1) ]

    """

    # parameter check
    features, groups, num_bins, edge_range, group_ids, num_groups, num_links = check_params(
            features, groups, num_bins, edge_range, trim_outliers, trim_percentile)

    weight_func, use_orig_distr, non_symmetric = check_weight_method(weight_method,
                                                                     use_original_distribution, asymmetric)

    # using the same bin edges for all nodes/groups to ensure correspondence
    # NOTE: common bin edges is important for the disances to be any meaningful
    edges = compute_bin_edges(features, num_bins, edge_range,
                              trim_outliers, trim_percentile, use_orig_distr)

    # handling special
    if relative_to_all:
        result = non_pairwise.relative_to_all(features, groups, edges, weight_func,
                                              use_orig_distr, group_ids, num_groups,
                                              return_networkx_graph, out_weights_path)
    else:
        result = pairwise_extract(features, groups, edges, weight_func, use_orig_distr,
                                  group_ids, num_groups, num_links,
                                  non_symmetric, return_networkx_graph, out_weights_path)

    # this can be a networkx graph or numpy array depending on request
    return result