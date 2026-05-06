def commit_config(self):
        """
        If replacement operation, perform 'configure replace' for the entire config.

        If merge operation, perform copy <file> running-config.
        """
        # Always generate a rollback config on commit
        self._gen_rollback_cfg()

        if self.config_replace:
            # Replace operation
            filename = self.candidate_cfg
            cfg_file = self._gen_full_path(filename)
            if not self._check_file_exists(cfg_file):
                raise ReplaceConfigException("Candidate config file does not exist")
            if self.auto_rollback_on_error:
                cmd = 'configure replace {} force revert trigger error'.format(cfg_file)
            else:
                cmd = 'configure replace {} force'.format(cfg_file)
            output = self._commit_hostname_handler(cmd)
            if ('original configuration has been successfully restored' in output) or \
               ('error' in output.lower()) or \
               ('failed' in output.lower()):
                msg = "Candidate config could not be applied\n{}".format(output)
                raise ReplaceConfigException(msg)
            elif '%Please turn config archive on' in output:
                msg = "napalm-ios replace() requires Cisco 'archive' feature to be enabled."
                raise ReplaceConfigException(msg)
        else:
            # Merge operation
            filename = self.merge_cfg
            cfg_file = self._gen_full_path(filename)
            if not self._check_file_exists(cfg_file):
                raise MergeConfigException("Merge source config file does not exist")
            cmd = 'copy {} running-config'.format(cfg_file)
            self._disable_confirm()
            output = self._commit_hostname_handler(cmd)
            self._enable_confirm()
            if 'Invalid input detected' in output:
                self.rollback()
                err_header = "Configuration merge failed; automatic rollback attempted"
                merge_error = "{0}:\n{1}".format(err_header, output)
                raise MergeConfigException(merge_error)

        # Save config to startup (both replace and merge)
        output += self.device.send_command_expect("write mem")