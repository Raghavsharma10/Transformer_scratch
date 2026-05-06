def load_buildfile(self, target):
        """Pull a build file from git."""
        log.info('Loading: %s', target)
        filepath = os.path.join(target.path, app.get_options().buildfile_name)
        try:
            repo = self.repo_state.GetRepo(target.repo)
            return repo.get_file(filepath)
        except gitrepo.GitError as err:
            log.error('Failed loading %s: %s', target, err)
            raise error.BrokenGraph('Sadface.')