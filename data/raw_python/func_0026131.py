def keypress(self, event):
        """Allow keys typed in widget to select items"""
        try:
            self.choice.set(self.shortcuts[event.keysym])
        except KeyError:
            # key not found (probably a bug, since we intend to catch
            # only events from shortcut keys, but ignore it anyway)
            pass