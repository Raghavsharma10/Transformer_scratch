def get_pull_command(self, remote=None, revision=None):
        """
        Get the command to pull changes from a remote repository into the local repository.

        When you pull a specific branch using git, the default behavior is to
        pull the change sets from the remote branch into the local repository
        and merge them into the *currently checked out* branch.

        What Mercurial does is to pull the change sets from the remote branch
        into the local repository and create a local branch whose contents
        mirror those of the remote branch. Merging is left to the operator.

        In my opinion the default behavior of Mercurial is more sane and
        predictable than the default behavior of git and so :class:`GitRepo`
        tries to emulate the default behavior of Mercurial.

        When a specific revision is pulled, the revision is assumed to be a
        branch name and git is instructed to pull the change sets from the
        remote branch into a local branch with the same name.

        .. warning:: The logic described above will undoubtedly break when
                     `revision` is given but is not a branch name. I'd fix
                     this if I knew how to, but I don't...
        """
        if revision:
            revision = '%s:%s' % (revision, revision)
        if self.bare:
            return [
                'git', 'fetch',
                remote or 'origin',
                # http://stackoverflow.com/a/10697486
                revision or '+refs/heads/*:refs/heads/*',
            ]
        else:
            command = ['git', 'pull']
            if remote or revision:
                command.append(remote or 'origin')
                if revision:
                    command.append(revision)
        return command