def write_metrics(baseName, values):
    """Write the metrics data

    :param baseName: The base name of the output files.
                     e.g. extensions will be appended to this base name
    :param values dictionary of values to write
    """
    m = open(baseName  + '_metrics.txt', 'w')
    for key in values:
        m.write(key + '=' + str(values[key]) + "\n")
    m.flush()
    m.close()