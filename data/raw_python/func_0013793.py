def _generate_tar(dir_path):
    """Private function that reads a local directory and generates a tar archive from it"""
    try:
        with tarfile.open(dir_path + '.tar', 'w') as tar:
            tar.add(dir_path)
    except tarfile.TarError as e:
        stderr("Error: tar archive creation failed [" + str(e) + "]", exit=1)