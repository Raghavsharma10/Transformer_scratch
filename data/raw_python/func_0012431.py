def write_stats_file(run_output_dict):
    """Writes a dummy PolyChord format .stats file for tests functions for
    processing stats files. This is written to:

    base_dir/file_root.stats

    Also returns the data in the file as a dict for comparison.

    Parameters
    ----------
    run_output_dict: dict
        Output information to write to .stats file. Must contain file_root and
        base_dir. If other settings are not specified, default values are used.

    Returns
    -------
    output: dict
        The expected output of
        nestcheck.process_polychord_stats(file_root, base_dir)
    """
    mandatory_keys = ['file_root', 'base_dir']
    for key in mandatory_keys:
        assert key in run_output_dict, key + ' not in run_output_dict'
    default_output = {'logZ': 0.0,
                      'logZerr': 0.0,
                      'logZs': [0.0],
                      'logZerrs': [0.0],
                      'ncluster': 1,
                      'nposterior': 0,
                      'nequals': 0,
                      'ndead': 0,
                      'nlike': 0,
                      'nlive': 0,
                      'avnlike': 0.0,
                      'avnlikeslice': 0.0,
                      'param_means': [0.0, 0.0, 0.0],
                      'param_mean_errs': [0.0, 0.0, 0.0]}
    allowed_keys = set(mandatory_keys) | set(default_output.keys())
    assert set(run_output_dict.keys()).issubset(allowed_keys), (
        'Input dict contains unexpected keys: {}'.format(
            set(run_output_dict.keys()) - allowed_keys))
    output = copy.deepcopy(run_output_dict)
    for key, value in default_output.items():
        if key not in output:
            output[key] = value
    # Make a PolyChord format .stats file corresponding to output
    file_lines = [
        'Evidence estimates:',
        '===================',
        ('  - The evidence Z is a log-normally distributed, with location and '
         'scale parameters mu and sigma.'),
        '  - We denote this as log(Z) = mu +/- sigma.',
        '',
        'Global evidence:',
        '----------------',
        '',
        'log(Z)       =  {0} +/-   {1}'.format(
            output['logZ'], output['logZerr']),
        '',
        '',
        'Local evidences:',
        '----------------',
        '']
    for i, (lz, lzerr) in enumerate(zip(output['logZs'], output['logZerrs'])):
        file_lines.append('log(Z_ {0})  =  {1} +/-   {2}'.format(
            str(i + 1).rjust(2), lz, lzerr))
    file_lines += [
        '',
        '',
        'Run-time information:',
        '---------------------',
        '',
        ' ncluster:          0 /       1',
        ' nposterior:        {0}'.format(output['nposterior']),
        ' nequals:           {0}'.format(output['nequals']),
        ' ndead:          {0}'.format(output['ndead']),
        ' nlive:             {0}'.format(output['nlive']),
        ' nlike:         {0}'.format(output['nlike']),
        ' <nlike>:       {0}   (    {1} per slice )'.format(
            output['avnlike'], output['avnlikeslice']),
        '',
        '',
        'Dim No.       Mean        Sigma']
    for i, (mean, meanerr) in enumerate(zip(output['param_means'],
                                            output['param_mean_errs'])):
        file_lines.append('{0}  {1} +/-   {2}'.format(
            str(i + 1).ljust(3), mean, meanerr))
    file_path = os.path.join(output['base_dir'],
                             output['file_root'] + '.stats')
    with open(file_path, 'w') as stats_file:
        stats_file.writelines('{}\n'.format(line) for line in file_lines)
    return output