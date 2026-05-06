def _new_ass_hierarchy(self, file_ass):
        """Returns a completely new cache hierarchy for given assistant file.

        Args:
             file_ass: the assistant from filesystem hierarchy to create cache hierarchy for
                      (for format see what refresh_role accepts)
        Returns:
            the newly created cache hierarchy
        """
        ret_struct = {'source': '',
                      'subhierarchy': {},
                      'attrs': {},
                      'snippets': {}}
        ret_struct['source'] = file_ass['source']
        self._ass_refresh_attrs(ret_struct, file_ass)

        for name, subhierarchy in file_ass['subhierarchy'].items():
            ret_struct['subhierarchy'][name] = self._new_ass_hierarchy(subhierarchy)

        return ret_struct