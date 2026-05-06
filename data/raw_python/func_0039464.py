def collect_outs(self):
        """Collect and store the outputs from this rule."""
        # TODO: this should probably live in CacheManager.
        for outfile in self.rule.output_files or []:
            outfile_built = os.path.join(self.buildroot, outfile)
            if not os.path.exists(outfile_built):
                raise error.TargetBuildFailed(
                    self.address, 'Output file is missing: %s' % outfile)

        #git_sha = gitrepo.RepoState().GetRepo(self.address.repo).repo.commit()
        # git_sha is insufficient, and is actually not all that useful.
        # More factors to include in hash:
        # - commit/state of source repo of all dependencies
        #   (or all input files?)
        #   - Actually I like that idea: hash all the input files!
        # - versions of build tools used (?)
        metahash = self._metahash()
        log.debug('[%s]: Metahash: %s', self.address, metahash.hexdigest())
        # TODO: record git repo state and buildoptions in cachemgr
        # TODO: move cachemgr to outer controller(?)
        self.cachemgr.putfile(outfile_built, self.buildroot, metahash)