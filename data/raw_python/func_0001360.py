def materialize_dict(bundle: dict, separator: str = '.') -> t.List[t.Tuple[str, t.Any]]:
    """
    Transforms a given ``bundle`` into a *sorted* list of tuples with materialized value paths and values:
    ``('path.to.value', <value>)``. Output is ordered by depth: the deepest element first.

    :param bundle: a dict to materialize
    :param separator: build paths with a given separator
    :return: a depth descending and alphabetically ascending sorted list (-deep, asc), the longest first

    ::

        sample = {
            'a': 1,
            'aa': 1,
            'b': {
                'c': 1,
                'b': 1,
                'a': 1,
                'aa': 1,
                'aaa': {
                    'a': 1
                }
            }
        }
        materialize_dict(sample, '/')
        [
            ('b/aaa/a', 1),
            ('b/a', 1),
            ('b/aa', 1),
            ('b/b', 1),
            ('b/c', 1),
            ('a', 1),
            ('aa', 1)
        ]
    """

    def _matkeysort(tup: t.Tuple[str, t.Any]):
        return len(tup[0].split(separator))

    s1 = sorted(_materialize_dict(bundle, separator=separator), key=lambda x: x[0])
    return sorted(s1, key=_matkeysort, reverse=True)