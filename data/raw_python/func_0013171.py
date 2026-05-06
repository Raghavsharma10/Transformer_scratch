def _check_file_exists(self, cfg_file):
        """
        Check that the file exists on remote device using full path.

        cfg_file is full path i.e. flash:/file_name

        For example
        # dir flash:/candidate_config.txt
        Directory of flash:/candidate_config.txt

        33  -rw-        5592  Dec 18 2015 10:50:22 -08:00  candidate_config.txt

        return boolean
        """
        cmd = 'dir {}'.format(cfg_file)
        success_pattern = 'Directory of {}'.format(cfg_file)
        output = self.device.send_command_expect(cmd)
        if 'Error opening' in output:
            return False
        elif success_pattern in output:
            return True
        return False