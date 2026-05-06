def get_actual_bp(self, lineno):
        """Get the actual breakpoint line number.

        When an exact match cannot be found in the lnotab expansion of the
        module code object or one of its subcodes, pick up the next valid
        statement line number.

        Return the statement line defined by the tuple (code firstlineno,
        statement line number) which is at the shortest distance to line
        'lineno' and greater or equal to 'lineno'. When 'lineno' is the first
        line number of a subcode, use its first statement line instead.
        """

        def _distance(code, module_level=False):
            """The shortest distance to the next valid statement."""
            subcodes = dict((c.co_firstlineno, c) for c in code.co_consts
                                if isinstance(c, types.CodeType) and not
                                    c.co_name.startswith('<'))
            # Get the shortest distance to the subcode whose first line number
            # is the last to be less or equal to lineno. That is, find the
            # index of the first subcode whose first_lno is the first to be
            # strictly greater than lineno.
            subcode_dist = None
            subcodes_flnos = sorted(subcodes)
            idx = bisect(subcodes_flnos, lineno)
            if idx != 0:
                flno = subcodes_flnos[idx-1]
                subcode_dist = _distance(subcodes[flno])

            # Check if lineno is a valid statement line number in the current
            # code, excluding function or method definition lines.
            code_lnos = sorted(code_line_numbers(code))
            # Do not stop at execution of function definitions.
            if not module_level and len(code_lnos) > 1:
                code_lnos = code_lnos[1:]
            if lineno in code_lnos and lineno not in subcodes_flnos:
                return 0, (code.co_firstlineno, lineno)

            # Compute the distance to the next valid statement in this code.
            idx = bisect(code_lnos, lineno)
            if idx == len(code_lnos):
                # lineno is greater that all 'code' line numbers.
                return subcode_dist
            actual_lno = code_lnos[idx]
            dist = actual_lno - lineno
            if subcode_dist and subcode_dist[0] < dist:
                return subcode_dist
            if actual_lno not in subcodes_flnos:
                return dist, (code.co_firstlineno, actual_lno)
            else:
                # The actual line number is the line number of the first
                # statement of the subcode following lineno (recursively).
                return _distance(subcodes[actual_lno])

        if self.code:
            code_dist = _distance(self.code, module_level=True)
        if not self.code or not code_dist:
            raise BdbSourceError('{}: line {} is after the last '
                'valid statement.'.format(self.filename, lineno))
        return code_dist[1]