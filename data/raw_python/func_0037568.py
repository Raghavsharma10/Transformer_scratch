def bit_for_bit(model_path, bench_path, config):
    """
    Checks whether the given files have bit for bit solution matches
    on the given variable list.

    Args:
        model_path: absolute path to the model dataset
        bench_path: absolute path to the benchmark dataset
        config: the configuration of the set of analyses

    Returns:
        A dictionary created by the elements object corresponding to
        the results of the bit for bit testing
    """
    fname = model_path.split(os.path.sep)[-1]
    # Error handling
    if not (os.path.isfile(bench_path) and os.path.isfile(model_path)):
        return elements.error("Bit for Bit",
                              "File named " + fname + " has no suitable match!")
    try:
        model_data = Dataset(model_path)
        bench_data = Dataset(bench_path)
    except (FileNotFoundError, PermissionError):
        return elements.error("Bit for Bit",
                              "File named " + fname + " could not be read!")
    if not (netcdf.has_time(model_data) and netcdf.has_time(bench_data)):
        return elements.error("Bit for Bit",
                              "File named " + fname + " could not be read!")

    # Begin bit for bit analysis
    headers = ["Max Error", "Index of Max Error", "RMS Error", "Plot"]
    stats = LIVVDict()
    for i, var in enumerate(config["bit_for_bit_vars"]):
        if var in model_data.variables and var in bench_data.variables:
            m_vardata = model_data.variables[var][:]
            b_vardata = bench_data.variables[var][:]
            diff_data = m_vardata - b_vardata
            if diff_data.any():
                stats[var]["Max Error"] = np.amax(np.absolute(diff_data))
                stats[var]["Index of Max Error"] = str(
                        np.unravel_index(np.absolute(diff_data).argmax(), diff_data.shape))
                stats[var]["RMS Error"] = np.sqrt(np.sum(np.square(diff_data).flatten()) /
                                                  diff_data.size)
                pf = plot_bit_for_bit(fname, var, m_vardata, b_vardata, diff_data)
            else:
                stats[var]["Max Error"] = stats[var]["RMS Error"] = 0
                pf = stats[var]["Index of Max Error"] = "N/A"
            stats[var]["Plot"] = pf
        else:
            stats[var] = {"Max Error": "No Match", "RMS Error": "N/A", "Plot": "N/A"}
    model_data.close()
    bench_data.close()
    return elements.bit_for_bit("Bit for Bit", headers, stats)