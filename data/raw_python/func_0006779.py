def add_usr_local_bin_to_path(log=False):
    """ adds /usr/local/bin to $PATH """
    if log:
        bookshelf2.logging_helpers.log_green('inserts /usr/local/bin into PATH')

    with settings(hide('warnings', 'running', 'stdout', 'stderr'),
                  capture=True):
        try:
            sudo('echo "export PATH=/usr/local/bin:$PATH" '
                 '|sudo /usr/bin/tee /etc/profile.d/fix-path.sh')
            return True
        except:
            raise SystemExit(1)