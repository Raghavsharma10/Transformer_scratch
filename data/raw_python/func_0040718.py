def _parse_scheme_file(self):
        """
        Initialize redundant data structures for lookup optimization
        """
        schemes = json.loads(self.scheme_file.read(), object_pairs_hook=OrderedDict)
        scheme_list = []
        scheme_dict = defaultdict(list)
        for scheme_len, scheme_group in schemes.items():
            for scheme_str, _count in scheme_group:
                scheme_code = tuple(int(c) for c in scheme_str.split(' '))
                scheme_list.append(scheme_code)
                scheme_dict[int(scheme_len)].append(len(scheme_list) - 1)
        return scheme_list, scheme_dict