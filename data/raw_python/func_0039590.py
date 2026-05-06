def dataset_dirpath(dataset_name=None, task=None, **kwargs):
    """Get the path of the corresponding dataset file.

    Parameters
    ----------
    dataset_name : str, optional
        The name of the dataset. Used to define a sub-directory to contain all
        instances of the dataset. If not given, a dataset-agnostic directory
        path is returned.
    task : str, optional
        The task for which datasets in the desired path are used for. If not
        given, a path for the corresponding task-agnostic directory is
        returned.
    **kwargs : extra keyword arguments
        Extra keyword arguments, representing additional attributes of the
        datasets, are used to determine additional sub-folders on the path.
        For example, providing 'lang=en' will results in a path such as
        '/barn_base_dir/regression/lang_en/mydataset.csv'. Hierarchy always
        matches lexicographical order of keyword argument names, so 'lang=en'
        and 'animal=dog' will result in a path such as
        'barn_base_dir/task_name/animal_dof/lang_en/dset.csv'.

    Returns
    -------
    str
        The path to the desired dataset file.
    """
    dataset_dir_path = data_dirpath(task=task, **kwargs)
    if dataset_name:
        dataset_dir_name = _snail_case(dataset_name)
        dataset_dir_path = os.path.join(dataset_dir_path, dataset_dir_name)
    os.makedirs(dataset_dir_path, exist_ok=True)
    return dataset_dir_path