def copy_default_config_to_user_directory(
        basename,
        clobber=False,
        dst_dir='~/.config/scriptabit'):
    """ Copies the default configuration file into the user config directory.

    Args:
        basename (str): The base filename.
        clobber (bool): If True, the default will be written even if a user
            config already exists.
        dst_dir (str): The destination directory.
    """
    dst_dir = os.path.expanduser(dst_dir)
    dst = os.path.join(dst_dir, basename)
    src = resource_filename(
        Requirement.parse("scriptabit"),
        os.path.join('scriptabit', basename))

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if clobber or not os.path.isfile(dst):
        shutil.copy(src, dst)