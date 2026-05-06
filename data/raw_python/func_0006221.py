def prepare_environment(work_dir):
        """
            Performs a few maintenance tasks before the Honeypot is run. Copies the data directory,
            and the config file to the cwd. The config file copied here is overwritten if
            the __init__ method is called with a configuration URL.

        :param work_dir: The directory to copy files to.
        """
        package_directory = os.path.dirname(os.path.abspath(beeswarm.__file__))

        logger.info('Copying data files to workdir.')
        shutil.copytree(os.path.join(package_directory, 'drones/honeypot/data'), os.path.join(work_dir, 'data/'),
                        ignore=Honeypot._ignore_copy_files)