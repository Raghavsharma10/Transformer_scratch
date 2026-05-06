def check_weight_method(weight_method_spec,
                        use_orig_distr=False,
                        allow_non_symmetric=False):
    "Check if weight_method is recognized and implemented, or ensure it is callable."

    if not isinstance(use_orig_distr, bool):
        raise TypeError('use_original_distribution flag must be boolean!')

    if not isinstance(allow_non_symmetric, bool):
        raise TypeError('allow_non_symmetric flag must be boolean')

    if isinstance(weight_method_spec, str):
        weight_method_spec = weight_method_spec.lower()

        if weight_method_spec in list_medpy_histogram_metrics:
            from medpy.metric import histogram as medpy_hist_metrics
            weight_func = getattr(medpy_hist_metrics, weight_method_spec)
            if use_orig_distr:
                warnings.warn('use_original_distribution must be False when using builtin histogram metrics, '
                                 'which expect histograms as input - setting it to False.', HiwenetWarning)
                use_orig_distr = False

        elif weight_method_spec in metrics_on_original_features:
            weight_func = getattr(more_metrics, weight_method_spec)
            if not use_orig_distr:
                warnings.warn('use_original_distribution must be True when using builtin non-histogram metrics, '
                              'which expect original feature values in ROI/node as input '
                              '- setting it to True.', HiwenetWarning)
                use_orig_distr = True

            if weight_method_spec in symmetric_metrics_on_original_features:
                print('Chosen metric is symmetric. Ignoring asymmetric=False flag.')
                allow_non_symmetric=False

        else:
            raise NotImplementedError('Chosen histogram distance/metric not implemented or invalid.')

    elif callable(weight_method_spec):
        # ensure 1) takes two ndarrays
        try:
            dummy_weight = weight_method_spec(make_random_histogram(), make_random_histogram())
        except:
            raise TypeError('Error applying given callable on two input arrays.\n'
                            '{} must accept two arrays and return a single scalar value!')
        else:
            # and 2) returns only one number
            if not np.isscalar(dummy_weight):
                raise TypeError('Given callable does not return a single scalar as output.')

        weight_func = weight_method_spec
    else:
        raise ValueError('Supplied method to compute edge weight is not recognized:\n'
                         'must be a string identifying one of the implemented methods\n{}'
                         '\n or a valid callable that accepts that two arrays '
                         'and returns 1 scalar.'.format(list_medpy_histogram_metrics))

    return weight_func, use_orig_distr, allow_non_symmetric