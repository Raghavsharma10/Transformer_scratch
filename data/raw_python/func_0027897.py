def focusin(self, event):
        """Change style on focus in events."""
        self.old_value = self.get()
        bc = self.style.lookup("TEntry", "bordercolor", ("focus",))
        dc = self.style.lookup("TEntry", "darkcolor", ("focus",))
        lc = self.style.lookup("TEntry", "lightcolor", ("focus",))
        self.style.configure("%s.spinbox.TFrame" % self.frame, bordercolor=bc,
                             darkcolor=dc, lightcolor=lc)