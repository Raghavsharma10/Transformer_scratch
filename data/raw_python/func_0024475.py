def _add_to_quick_menu(self, key, wf):
        """
        Appends menu entries to dashboard quickmenu according
        to :attr:`zengine.settings.QUICK_MENU`

        Args:
            key: workflow name
            wf: workflow menu entry
        """
        if key in settings.QUICK_MENU:
            self.output['quick_menu'].append(wf)