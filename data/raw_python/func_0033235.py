def _get_result_paths(self,data):
        """ Get the resulting tree"""
        result = {}
        result['Tree'] = ResultPath(Path=splitext(self._input_filename)[0] + \
                                                  '.tree')
        return result