def color_str(self):
        "Return an escape-coded string to write to the terminal."
        s = self.s
        for k, v in sorted(self.atts.items()):
            # (self.atts sorted for the sake of always acting the same.)
            if k not in xforms:
                # Unsupported SGR code
                continue
            elif v is False:
                continue
            elif v is True:
                s = xforms[k](s)
            else:
                s = xforms[k](s, v)
        return s