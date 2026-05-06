def _cmp_models(self, m1, m2):
        """Compare two models from different swagger APIs and tell if they are
        equal (return 0), or not (return != 0)"""

        # Don't alter m1/m2 by mistake
        m1 = copy.deepcopy(m1)
        m2 = copy.deepcopy(m2)

        # Remove keys added by bravado-core
        def _cleanup(d):
            """Remove all keys in the blacklist"""
            for k in ('x-model', 'x-persist', 'x-scope'):
                if k in d:
                    del d[k]
            for v in list(d.values()):
                if isinstance(v, dict):
                    _cleanup(v)

        _cleanup(m1)
        _cleanup(m2)

        # log.debug("model1:\n" + pprint.pformat(m1))
        # log.debug("model2:\n" + pprint.pformat(m2))
        return not m1 == m2