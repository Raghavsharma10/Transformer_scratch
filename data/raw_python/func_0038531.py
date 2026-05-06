def weak_scaling(timing_stats, scaling_var, data_points):
    """
    Generate data for plotting weak scaling.  The data points keep
    a constant amount of work per processor for each data point.

    Args:
        timing_stats: the result of the generate_timing_stats function
        scaling_var: the variable to select from the timing_stats dictionary
                     (can be provided in configurations via the 'scaling_var' key)
        data_points: the list of size and processor counts to use as data
                     (can be provided in configurations via the 'weak_scaling_points' key)

    Returns:
         A dict of the form:
            {'bench' : {'mins' : [], 'means' : [], 'maxs' : []},
             'model' : {'mins' : [], 'means' : [], 'maxs' : []},
             'proc_counts' : []}
    """
    timing_data = dict()
    proc_counts = []
    bench_means = []
    bench_mins = []
    bench_maxs = []
    model_means = []
    model_mins = []
    model_maxs = []
    for point in data_points:
        size = point[0]
        proc = point[1]
        try:
            model_data = timing_stats[size][proc]['model'][scaling_var]
            bench_data = timing_stats[size][proc]['bench'][scaling_var]
        except KeyError:
            continue
        proc_counts.append(proc)
        model_means.append(model_data['mean'])
        model_mins.append(model_data['min'])
        model_maxs.append(model_data['max'])
        bench_means.append(bench_data['mean'])
        bench_mins.append(bench_data['min'])
        bench_maxs.append(bench_data['max'])
    timing_data['bench'] = dict(mins=bench_mins, means=bench_means, maxs=bench_maxs)
    timing_data['model'] = dict(mins=model_mins, means=model_means, maxs=model_maxs)
    timing_data['proc_counts'] = [int(pc[1:]) for pc in proc_counts]
    return timing_data