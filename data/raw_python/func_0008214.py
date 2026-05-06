def show(self, msg, indent=0, style="", **kwargs):
        """
        Print message to console, indent format may apply.
        """
        if self.enable_verbose:
            new_msg = self.MessageTemplate.with_style.format(
                indent=self.tab * indent,
                style=style,
                msg=msg,
            )
            print(new_msg, **kwargs)