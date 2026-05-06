def load_replace_candidate(self, filename=None, config=None):
        """
        SCP file to device filesystem, defaults to candidate_config.

        Return None or raise exception
        """
        self.config_replace = True
        return_status, msg = self._load_candidate_wrapper(source_file=filename,
                                                          source_config=config,
                                                          dest_file=self.candidate_cfg,
                                                          file_system=self.dest_file_system)
        if not return_status:
            raise ReplaceConfigException(msg)