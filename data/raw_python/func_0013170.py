def _gen_rollback_cfg(self):
        """Save a configuration that can be used for rollback."""
        cfg_file = self._gen_full_path(self.rollback_cfg)
        cmd = 'copy running-config {}'.format(cfg_file)
        self._disable_confirm()
        self.device.send_command_expect(cmd)
        self._enable_confirm()