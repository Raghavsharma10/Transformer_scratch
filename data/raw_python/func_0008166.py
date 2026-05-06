def do(self, **kwargs):
        """
        Here for compatibility with legacy clients only - DO NOT USE!!!
        This is sort of mix of "append" and "insert": it puts commands in the list,
        with some half smarts about which commands go at the front or back.
        If you add multiple commands to the back in one call, they will get added sorted by command name.
        :param kwargs: the commands in key=val format
        :return: the Action, so you can do Action(...).do(...).do(...)
        """
        # add "create" / "add" / "removeFrom" first
        for k, v in list(six.iteritems(kwargs)):
            if k.startswith("create") or k.startswith("addAdobe") or k.startswith("removeFrom"):
                self.commands.append({k: v})
                del kwargs[k]

        # now do the other actions, in a canonical order (to avoid py2/py3 variations)
        for k, v in sorted(six.iteritems(kwargs)):
            if k in ['add', 'remove']:
                self.commands.append({k: {"product": v}})
            else:
                self.commands.append({k: v})
        return self