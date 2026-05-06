def fetch_svn(self, svn_uri, directory):
        """
        Fetch subversion repository

        @param svn_uri: subversion repository uri to check out
        @type svn_uri: string

        @param directory: directory to download to
        @type directory: string

        @returns: 0 = success or 1 for failed download


        """
        if not command_successful("svn --version"):
            self.logger.error("ERROR: Do you have subversion installed?")
            return 1
        if os.path.exists(directory):
            self.logger.error("ERROR: Checkout directory exists - %s" \
                    % directory)
            return 1
        try:
            os.mkdir(directory)
        except OSError as err_msg:
            self.logger.error("ERROR: " + str(err_msg))
            return 1
        cwd = os.path.realpath(os.curdir)
        os.chdir(directory)
        self.logger.info("Doing subversion checkout for %s" % svn_uri)
        status, output = run_command("/usr/bin/svn co %s" % svn_uri)
        self.logger.info(output)
        os.chdir(cwd)
        self.logger.info("subversion checkout is in directory './%s'" \
                % directory)
        return 0