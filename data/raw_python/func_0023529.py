def bp_commands(self, frame, breakpoint_hits):
        """Call every command that was set for the current active breakpoints.

        Returns True if the normal interaction function must be called,
        False otherwise."""
        # Handle multiple breakpoints on the same line (issue 14789)
        effective_bp_list, temporaries = breakpoint_hits
        silent = True
        doprompt = False
        atleast_one_cmd = False
        for bp in effective_bp_list:
            if bp in self.commands:
                if not atleast_one_cmd:
                    atleast_one_cmd = True
                    self.setup(frame, None)
                lastcmd_back = self.lastcmd
                for line in self.commands[bp]:
                    self.onecmd(line)
                self.lastcmd = lastcmd_back
                if not self.commands_silent[bp]:
                    silent = False
                if self.commands_doprompt[bp]:
                    doprompt = True
        # Delete the temporary breakpoints.
        tmp_to_delete = ' '.join(str(bp) for bp in temporaries)
        if tmp_to_delete:
            self.do_clear(tmp_to_delete)

        if atleast_one_cmd:
            return doprompt, silent
        return None