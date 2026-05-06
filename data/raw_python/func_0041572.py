def clone_data(self, data_path):
        """
        Clones data for given data_path:
        :param str data_path: Git url (git/http/https) or local directory path
        """
        self.data_path = data_path

        data_url = urlparse.urlparse(self.data_path)
        if data_url.scheme in SCHEMES or (data_url.scheme == '' and ':' in data_url.path):
            data_name = os.path.splitext(os.path.basename(data_url.path))[0]
            data_destination = os.path.join(self.clone_dir, data_name)
            clone_data = True
            if os.path.isdir(data_destination):
                self.logger.info('Data clone directory already exists, checking commit sha')
                with Dir(data_destination):
                    # check the current status of what's local
                    rc, out, err = self.cmd.gather("git status -sb")
                    if rc:
                        raise GitDataException('Error getting data repo status: {}'.format(err))

                    lines = out.strip().split('\n')
                    synced = ('ahead' not in lines[0] and 'behind' not in lines[0] and len(lines) == 1)

                    # check if there are unpushed
                    # verify local branch
                    rc, out, err = self.cmd.gather("git rev-parse --abbrev-ref HEAD")
                    if rc:
                        raise GitDataException('Error checking local branch name: {}'.format(err))
                    branch = out.strip()
                    if branch != self.branch:
                        if not synced:
                            msg = ('Local branch is `{}`, but requested `{}` and you have uncommitted/pushed changes\n'
                                   'You must either clear your local data or manually checkout the correct branch.'
                                   ).format(branch, self.branch)
                            raise GitDataBranchException(msg)
                    else:
                        # Check if local is synced with remote
                        rc, out, err = self.cmd.gather(["git", "ls-remote", self.data_path, self.branch])
                        if rc:
                            raise GitDataException('Unable to check remote sha: {}'.format(err))
                        remote = out.strip().split('\t')[0]
                        try:
                            self.cmd.check_assert('git branch --contains {}'.format(remote))
                            self.logger.info('{} is already cloned and latest'.format(self.data_path))
                            clone_data = False
                        except:
                            if not synced:
                                msg = ('Local data is out of sync with remote and you have unpushed commits: {}\n'
                                       'You must either clear your local data\n'
                                       'or manually rebase from latest remote to continue'
                                       ).format(data_destination)
                                raise GitDataException(msg)

            if clone_data:
                if os.path.isdir(data_destination):  # delete if already there
                    shutil.rmtree(data_destination)
                self.logger.info('Cloning config data from {}'.format(self.data_path))
                if not os.path.isdir(data_destination):
                    cmd = "git clone -b {} --depth 1 {} {}".format(self.branch, self.data_path, data_destination)
                    rc, out, err = self.cmd.gather(cmd)
                    if rc:
                        raise GitDataException('Error while cloning data: {}'.format(err))

            self.remote_path = self.data_path
            self.data_path = data_destination
        elif data_url.scheme in ['', 'file']:
            self.remote_path = None
            self.data_path = os.path.abspath(self.data_path)  # just in case relative path was given
        else:
            raise ValueError(
                'Invalid data_path: {} - invalid scheme: {}'
                .format(self.data_path, data_url.scheme)
            )

        if self.sub_dir:
            self.data_dir = os.path.join(self.data_path, self.sub_dir)
        else:
            self.data_dir = self.data_path
        if not os.path.isdir(self.data_dir):
            raise GitDataPathException('{} is not a valid sub-directory in the data'.format(self.sub_dir))