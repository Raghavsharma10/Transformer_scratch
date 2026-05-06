def _get_log_commits(self):
        """
        calls git log to complile a change list
        """
        ## check if update is necessary
        cmd = "git log --pretty=oneline {}..".format(self.init_version)
        cmdlist = shlex.split(cmd)
        commits = subprocess.check_output(cmdlist)
        
        ## Split off just the first element, we don't need commit tag
        self.commits = [x.split(" ", 1) for x in commits.split("\n")]