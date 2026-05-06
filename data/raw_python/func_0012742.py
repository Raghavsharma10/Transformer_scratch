def prepare_run(run_id, run_data):
    """Prepare run inputs for one of multiple EnergyPlus runs.

    :param run_id: An ID number for naming the IDF.
    :param run_data: Tuple of the IDF and keyword args to pass to EnergyPlus executable.
    :return: Tuple of the IDF path and EPW, and the keyword args.
    """
    idf, kwargs = run_data
    epw = idf.epw
    idf_dir = os.path.join('multi_runs', 'idf_%i' % run_id)
    os.mkdir(idf_dir)
    idf_path = os.path.join(idf_dir, 'in.idf')
    idf.saveas(idf_path)
    return (idf_path, epw), kwargs