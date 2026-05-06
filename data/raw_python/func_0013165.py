def _commit_hostname_handler(self, cmd):
        """Special handler for hostname change on commit operation."""
        current_prompt = self.device.find_prompt().strip()
        terminating_char = current_prompt[-1]
        pattern = r"[>#{}]\s*$".format(terminating_char)
        # Look exclusively for trailing pattern that includes '#' and '>'
        output = self.device.send_command_expect(cmd, expect_string=pattern)
        # Reset base prompt in case hostname changed
        self.device.set_base_prompt()
        return output