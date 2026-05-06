def parse_gptl(file_path, var_list):
    """
    Read a GPTL timing file and extract some data.

    Args:
        file_path: the path to the GPTL timing file
        var_list: a list of strings to look for in the file

    Returns:
        A dict containing key-value pairs of the livvkit
        and the times associated with them
    """
    timing_result = dict()
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            for var in var_list:
                for line in f:
                    if var in line:
                        timing_result[var] = float(line.split()[4])/int(line.split()[2])
                        break
    return timing_result