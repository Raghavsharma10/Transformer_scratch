def calculate_offset(cls, labels):
        '''Return the maximum length of the provided strings that have a nice
        variant in DapFormatter._nice_strings'''
        used_strings = set(cls._nice_strings.keys()) & set(labels)
        return max([len(cls._nice_strings[s]) for s in used_strings])