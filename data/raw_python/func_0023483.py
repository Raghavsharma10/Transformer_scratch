def process_hit_event(self, frame):
        """Return (stop_state, delete_temporary) at a breakpoint hit event."""
        if not self.enabled:
            return False, False
        # Count every hit when breakpoint is enabled.
        self.hits += 1
        # A conditional breakpoint.
        if self.cond:
            try:
                if not eval_(self.cond, frame.f_globals, frame.f_locals):
                    return False, False
            except Exception:
                # If the breakpoint condition evaluation fails, the most
                # conservative thing is to stop on the breakpoint.  Don't
                # delete temporary, as another hint to the user.
                return True, False
        if self.ignore > 0:
            self.ignore -= 1
            return False, False
        return True, True