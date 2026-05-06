def find_path(self, in_, out):
        '''
        Given an input and output TypeString, produce a graph traversal,
        keeping in mind special options like Conversion Profiles, Preferred
        Paths, and Direct Conversions.
        '''
        if in_.arguments:
            raise ValueError('Cannot originate path in argumented TypeString')

        # Determine conversion profile. This is either simply the output, OR,
        # if a custom profile has been specified for this output, that custom
        # path or type is used.
        profile = self.conversion_profiles.get(str(out), str(out))
        if isinstance(profile, str):
            profile = (profile, )
        types_by_format = {_format(s): TypeString(s) for s in profile}

        # Normalize input and output types to string
        in_str = str(in_)
        out_str = _format(profile[0])

        # First check for direct conversions, returning immediately if found
        direct_converter = self.direct_converters.get((in_str, out_str))
        if direct_converter:
            out_ts = types_by_format.get(out_str, TypeString(out_str))
            return [(direct_converter, TypeString(in_str), out_ts)]

        # No direct conversions was found, so find path through graph.
        # If profile was plural, add in extra steps.
        path = self.dgraph.shortest_path(in_str, out_str)
        path += profile[1:]

        # Loop through each edge traversal, adding converters and type
        # string pairs as we go along. This is to ensure conversion
        # profiles that have arguments mid-profile get included.
        results = []
        for left, right in pair_looper(path):
            converter = self.converters.get((_format(left), _format(right)))
            right_typestring = types_by_format.get(right, TypeString(right))
            results.append((converter, TypeString(left), right_typestring))
        return results