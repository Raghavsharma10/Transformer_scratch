def replace_folder(path):
    """If the specified folder exists, it is deleted and recreated"""
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)