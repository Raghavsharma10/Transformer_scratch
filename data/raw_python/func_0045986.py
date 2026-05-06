def return_logfile(filename, log_dir='/var/log'):
        """Return a path for logging file.

        If ``log_dir`` exists and the userID is 0 the log file will be written
        to the provided log directory. If the UserID is not 0 or log_dir does
        not exist the log file will be written to the users home folder.

        :param filename: ``str``
        :param log_dir: ``str``
        :return: ``str``
        """
        if sys.platform == 'win32':
            user = getpass.getuser()
        else:
            user = os.getuid()
        home = os.path.expanduser('~')

        if not os.path.isdir(log_dir):
            return os.path.join(home, filename)

        log_dir_stat = os.stat(log_dir)
        if log_dir_stat.st_uid == user:
            return os.path.join(log_dir, filename)
        elif log_dir_stat.st_gid == user:
            return os.path.join(log_dir, filename)
        else:
            return os.path.join(home, filename)