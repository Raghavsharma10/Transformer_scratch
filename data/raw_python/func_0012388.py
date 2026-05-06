def process_polychord_stats(file_root, base_dir):
    """Reads a PolyChord <root>.stats output file and returns the information
    contained in a dictionary.

    Parameters
    ----------
    file_root: str
        Root for run output file names (PolyChord file_root setting).
    base_dir: str
        Directory containing data (PolyChord base_dir setting).

    Returns
    -------
    output: dict
        See PolyChord documentation for more details.
    """
    filename = os.path.join(base_dir, file_root) + '.stats'
    output = {'base_dir': base_dir,
              'file_root': file_root}
    with open(filename, 'r') as stats_file:
        lines = stats_file.readlines()
    output['logZ'] = float(lines[8].split()[2])
    output['logZerr'] = float(lines[8].split()[4])
    # Cluster logZs and errors
    output['logZs'] = []
    output['logZerrs'] = []
    for line in lines[14:]:
        if line[:5] != 'log(Z':
            break
        output['logZs'].append(float(
            re.findall(r'=(.*)', line)[0].split()[0]))
        output['logZerrs'].append(float(
            re.findall(r'=(.*)', line)[0].split()[2]))
    # Other output info
    nclust = len(output['logZs'])
    output['ncluster'] = nclust
    output['nposterior'] = int(lines[20 + nclust].split()[1])
    output['nequals'] = int(lines[21 + nclust].split()[1])
    output['ndead'] = int(lines[22 + nclust].split()[1])
    output['nlive'] = int(lines[23 + nclust].split()[1])
    try:
        output['nlike'] = int(lines[24 + nclust].split()[1])
    except ValueError:
        # if nlike has too many digits, PolyChord just writes ***** to .stats
        # file. This causes a ValueError
        output['nlike'] = np.nan
    output['avnlike'] = float(lines[25 + nclust].split()[1])
    output['avnlikeslice'] = float(lines[25 + nclust].split()[3])
    # Means and stds of dimensions (not produced by PolyChord<=1.13)
    if len(lines) > 29 + nclust:
        output['param_means'] = []
        output['param_mean_errs'] = []
        for line in lines[29 + nclust:]:
            output['param_means'].append(float(line.split()[1]))
            output['param_mean_errs'].append(float(line.split()[3]))
    return output