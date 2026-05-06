def _get_id(self):
        """Construct and return the identifier"""
        return ''.join(map(str,
                           filter(is_not_None,
                                  [self.Prefix, self.Name])))