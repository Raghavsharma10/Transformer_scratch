def _getPattern(self, ipattern, done=None):
        """Parses sort pattern.

        :ipattern: A pattern to parse.
        :done:  If :ipattern: refers to done|undone,
        use this to indicate proper state.
        :returns: A pattern suitable for Model.modify.

        """
        if ipattern is None:
            return None
        if ipattern is True:
            if done is not None:
                return ([(None, None, done)], {})
            # REMEMBER: This False is for sort reverse!
            return ([(0, False)], {})

        def _getReverse(pm):
            return pm == '-'

        def _getIndex(k):
            try:
                return int(k)
            except ValueError:
                raise InvalidPatternError(k, "Invalid level number")

        def _getDone(p):
            v = p.split('=')
            if len(v) == 2:
                try:
                    return (Model.indexes[v[0]], v[1], done)
                except KeyError:
                    raise InvalidPatternError(v[0], 'Invalid field name')
            return (None, v[0], done)
        ipattern1 = list()
        ipattern2 = dict()
        for s in ipattern.split(','):
            if done is not None:
                v = done
            else:
                v = _getReverse(s[-1])
            k = s.split(':')
            if len(k) == 1:
                if done is not None:
                    ipattern1.append(_getDone(k[0]))
                    continue
                ko = k[0][:-1]
                try:
                    if len(k[0]) == 1:
                        k = 0
                    else:
                        k = Model.indexes[ko]
                except KeyError:
                    k = _getIndex(k[0][:-1])
                else:
                    ipattern1.append((k, v))
                    continue
                v = (0, v)
            elif len(k) == 2:
                try:
                    if done is not None:
                        v = _getDone(k[1])
                    else:
                        v = (Model.indexes[k[1][:-1]], v)
                    k = _getIndex(k[0])
                except KeyError:
                    raise InvalidPatternError(k[1][:-1], 'Invalid field name')
            else:
                raise InvalidPatternError(s, 'Unrecognized token in')
            ipattern2.setdefault(k, []).append(v)
        return (ipattern1, ipattern2)