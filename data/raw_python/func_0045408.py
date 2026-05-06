def code(self):
        """
        code
        """

        def uniq(seq):
            """
            @type seq: str
            @return: None
            """
            seen = set()
            seen_add = seen.add
            return [x for x in seq if x not in seen and not seen_add(x)]

        # noinspection PyTypeChecker
        a = uniq(i for i in self.autos if i is not None)
        # noinspection PyTypeChecker
        e = uniq(i for i in self.errors if i is not None)

        if e:
            return '\n'.join(e)

        return '\n'.join(a)