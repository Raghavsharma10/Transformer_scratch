def user_line(self, frame, breakpoint_hits=None):
        """This function is called when we stop or break at this line."""
        if not breakpoint_hits:
            self.interaction(frame, None)
        else:
            commands_result = self.bp_commands(frame, breakpoint_hits)
            if not commands_result:
                self.interaction(frame, None)
            else:
                doprompt, silent = commands_result
                if not silent:
                    self.print_stack_entry(self.stack[self.curindex])
                if doprompt:
                    self._cmdloop()
                self.forget()