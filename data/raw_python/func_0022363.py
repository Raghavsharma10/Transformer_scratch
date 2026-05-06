def clear_cache(dataset_name=None):
    """Remove a data set from the cache"""
    dr = data_resources[dataset_name]
    if 'dirs' in dr:
        for dirs, files in zip(dr['dirs'], dr['files']):
            for dir, file in zip(dirs, files):
                path = os.path.join(data_path, dataset_name, dir, file)
                if os.path.exists(path):
                    logging.info("clear_cache: removing " + path)
                    os.unlink(path)
            for dir in dirs:
                path = os.path.join(data_path, dataset_name, dir)
                if os.path.exists(path):
                    logging.info("clear_cache: remove directory " + path)
                    os.rmdir(path)
        
    else:
        for file_list in dr['files']:
            for file in file_list:
                path = os.path.join(data_path, dataset_name, file)
                if os.path.exists(path):
                    logging.info("clear_cache: remove " + path)
                    os.unlink(path)