def diff_configurations(model_config, bench_config, model_bundle, bench_bundle):
    """
    Description

    Args:
        model_config: a dictionary with the model configuration data
        bench_config: a dictionary with the benchmark configuration data
        model_bundle: a LIVVkit model bundle object
        bench_bundle: a LIVVkit model bundle object

    Returns:
        A dictionary created by the elements object corresponding to
        the results of the bit for bit testing
    """
    diff_dict = LIVVDict()
    model_data = model_bundle.parse_config(model_config)
    bench_data = bench_bundle.parse_config(bench_config)
    if model_data == {} and bench_data == {}:
        return elements.error("Configuration Comparison",
                              "Could not open file: " + model_config.split(os.path.sep)[-1])

    model_sections = set(six.iterkeys(model_data))
    bench_sections = set(six.iterkeys(bench_data))
    all_sections = set(model_sections.union(bench_sections))

    for s in all_sections:
        model_vars = set(six.iterkeys(model_data[s])) if s in model_sections else set()
        bench_vars = set(six.iterkeys(bench_data[s])) if s in bench_sections else set()
        all_vars = set(model_vars.union(bench_vars))
        for v in all_vars:
            model_val = model_data[s][v] if s in model_sections and v in model_vars else 'NA'
            bench_val = bench_data[s][v] if s in bench_sections and v in bench_vars else 'NA'
            same = True if model_val == bench_val and model_val != 'NA' else False
            diff_dict[s][v] = (same, model_val, bench_val)
    return elements.file_diff("Configuration Comparison", diff_dict)