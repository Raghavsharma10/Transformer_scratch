def remove_tar_files(file_list):
    """Public function that removes temporary tar archive files in a local directory"""
    for f in file_list:
        if file_exists(f) and f.endswith('.tar'):
            os.remove(f)