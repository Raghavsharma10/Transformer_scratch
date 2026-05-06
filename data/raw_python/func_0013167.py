def discard_config(self):
        """Set candidate_cfg to current running-config. Erase the merge_cfg file."""
        discard_candidate = 'copy running-config {}'.format(self._gen_full_path(self.candidate_cfg))
        discard_merge = 'copy null: {}'.format(self._gen_full_path(self.merge_cfg))
        self._disable_confirm()
        self.device.send_command_expect(discard_candidate)
        self.device.send_command_expect(discard_merge)
        self._enable_confirm()