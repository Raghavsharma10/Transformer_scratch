def fetch_artifact(self, trial_id, prefix):
        """
        Verifies that all children of the artifact prefix path are
        available locally. Fetches them if not.

        Returns the local path to the given trial's artifacts at the
        specified prefix, which is always just

        {log_dir}/{trial_id}/{prefix}
        """
        # TODO: general windows concern: local prefix will be in
        # backslashes but remote dirs will be expecting /
        # TODO: having s3 logic split between project and sync.py
        # worries me
        local = os.path.join(self.log_dir, trial_id, prefix)
        if self.upload_dir:
            remote = '/'.join([self.upload_dir, trial_id, prefix])
            _remote_to_local_sync(remote, local)
        return local