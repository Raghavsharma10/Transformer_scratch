def data_available(dataset_name=None):
    """Check if the data set is available on the local machine already."""
    dr = data_resources[dataset_name]
    if 'dirs' in dr:
        for dirs, files in zip(dr['dirs'], dr['files']):
            for dir, file in zip(dirs, files):
                if not os.path.exists(os.path.join(data_path, dataset_name, dir, file)):
                    return False
    else:
        for file_list in dr['files']:
            for file in file_list:
                if not os.path.exists(os.path.join(data_path, dataset_name, file)):
                    return False
    return True