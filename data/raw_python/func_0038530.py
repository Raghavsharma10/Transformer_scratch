def generate_timing_stats(file_list, var_list):
    """
    Parse all of the timing files, and generate some statistics
    about the run.

    Args:
        file_list: A list of timing files to parse
        var_list: A list of variables to look for in the timing file

    Returns:
        A dict containing values that have the form:
            [mean, min, max, mean, standard deviation]
    """
    timing_result = dict()
    timing_summary = dict()
    for file in file_list:
        timing_result[file] = functions.parse_gptl(file, var_list)
    for var in var_list:
        var_time = []
        for f, data in timing_result.items():
            try:
                var_time.append(data[var])
            except:
                continue
        if len(var_time):
            timing_summary[var] = {'mean': np.mean(var_time),
                                   'max': np.max(var_time),
                                   'min': np.min(var_time),
                                   'std': np.std(var_time)}
    return timing_summary