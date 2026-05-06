def _delete_file(configurator, path):
    """ remove file and remove it's directories if empty """
    path = os.path.join(configurator.target_directory, path)
    os.remove(path)
    try:
        os.removedirs(os.path.dirname(path))
    except OSError:
        pass