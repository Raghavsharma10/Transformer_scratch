def update_label(self, name, color, new_name=''):
        """Update the label ``name``.

        :param str name: (required), name of the label
        :param str color: (required), color code
        :param str new_name: (optional), new name of the label
        :returns: bool
        """
        label = self.label(name)
        resp = False
        if label:
            upd = label.update
            resp = upd(new_name, color) if new_name else upd(name, color)
        return resp