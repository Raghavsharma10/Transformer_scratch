def _apply_nested_privacy(self, data):
        """ Apply privacy to nested documents.

        :param data: Dict of data to which privacy is already applied.
        """
        kw = {
            'is_admin': self.is_admin,
            'drop_hidden': self.drop_hidden,
        }
        for key, val in data.items():
            if is_document(val):
                data[key] = apply_privacy(self.request)(result=val, **kw)
            elif isinstance(val, list) and val and is_document(val[0]):
                data[key] = [apply_privacy(self.request)(result=doc, **kw)
                             for doc in val]
        return data