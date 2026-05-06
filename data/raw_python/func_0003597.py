def _merge_dicts(self, dict1, dict2, path=[]):
        "merges dict2 into dict1"
        for key in dict2:
            if key in dict1:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    self._merge_dicts(dict1[key], dict2[key], path + [str(key)])
                elif dict1[key] != dict2[key]:
                    if self._options.get('allowmultioptions', True):
                        if not isinstance(dict1[key], list):
                            dict1[key] = [dict1[key]]
                        if not isinstance(dict2[key], list):
                            dict2[key] = [dict2[key]]
                        dict1[key] = self._merge_lists(dict1[key], dict2[key])
                    else:
                        if self._options.get('mergeduplicateoptions', False):
                            dict1[key] = dict2[key]
                        else:
                            raise error.ApacheConfigError('Duplicate option "%s" prohibited' % '.'.join(path + [str(key)]))
            else:
                dict1[key] = dict2[key]
        return dict1