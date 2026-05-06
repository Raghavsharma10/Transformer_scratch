def candidates(self):
        """A list of candidate addresses (as dictionaries) from a geocode
           operation"""
        # convert x['location'] to a point from a json point struct
        def cditer():
            for candidate in self._json_struct['candidates']:
                newcandidate = candidate.copy()
                newcandidate['location'] = \
                    geometry.fromJson(newcandidate['location'])
                yield newcandidate
        return list(cditer())