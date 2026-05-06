def squash(self, a, b):
        """
        Returns a generator that squashes two iterables into one.

        ```
        ['this', 'that'], [[' and', ' or']] => ['this and', 'this or', 'that and', 'that or']
        ```
        """

        return ((''.join(x) if isinstance(x, tuple) else x) for x in itertools.product(a, b))