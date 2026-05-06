def _operators_replace(self, string: str) -> str:
        """
        Searches for first unary or binary operator (via self.op_regex
        that has only one group that contain operator)
        then replaces it (or escapes it if brackets do not match).
        Everything until:
          * space ' '
          * begin/end of the string
          * bracket from outer scope (like '{a/b}': term1=a term2=b)
        is considered a term (contents of matching brackets '{}' are
        ignored).

        Attributes
        ----------
        string: str
            string to replace
        """
        # noinspection PyShadowingNames
        def replace(string: str, start: int, end: int, substring: str) -> str:
            return string[0:start] + substring + string[end:len(string)]

        # noinspection PyShadowingNames
        def sub_pat(pat: Callable[[list], str] or str, terms: list) -> str:
            if isinstance(pat, str):
                return pat.format(*terms)
            else:
                return pat(terms)

        count = 0

        def check():
            nonlocal count
            count += 1
            if count > self.max_while:
                raise RuntimeError('Presumably while loop is stuck')

        # noinspection PyShadowingNames
        def null_replace(match) -> str:
            regex_terms = [gr for gr in match.groups() if gr is not None]
            op = regex_terms[0]
            terms = regex_terms[1:]
            return sub_pat(self.null_ops.ops[op]['pat'], terms)

        string = self.null_ops.regex.sub(null_replace, string)

        for ops, loc in [(self.pref_un_ops, 'r'), (self.postf_un_ops, 'l'),
                         (self.bin_centr_ops, 'lr')]:
            count = 0
            match = ops.regex.search(string)
            while match:
                check()
                regex_terms = [gr for gr in match.groups() if gr is not None]
                op = regex_terms[0]
                loc_map = self._local_map(match, loc)
                lmatch, rmatch = None, None
                if loc == 'l' or loc == 'lr':
                    for m in ops.ops[op]['pref'].finditer(string):
                        if m.end() <= match.start() and loc_map[m.end() - 1] == 0:
                            lmatch = m
                    if lmatch is None:
                        string = replace(string, match.start(), match.end(), match.group(0).replace(op, '\\' + op))
                        match = ops.regex.search(string)
                        continue
                    else:
                        term1 = string[lmatch.end():match.start()]
                if loc == 'r' or loc == 'lr':
                    for m in ops.ops[op]['postf'].finditer(string):
                        if m.start() >= match.end() and loc_map[m.start()] == 0:
                            rmatch = m
                            break
                    if rmatch is None:
                        string = replace(string, match.start(), match.end(), match.group(0).replace(op, '\\' + op))
                        match = ops.regex.search(string)
                        continue
                    else:
                        term2 = string[match.end():rmatch.start()]
                if loc == 'l':
                    # noinspection PyUnboundLocalVariable
                    terms = list(lmatch.groups()) + [term1] + regex_terms[1:]
                    start, end = lmatch.start(), match.end()
                elif loc == 'r':
                    # noinspection PyUnboundLocalVariable
                    terms = regex_terms[1:] + [term2] + list(rmatch.groups())
                    start, end = match.start(), rmatch.end()
                elif loc == 'lr':
                    terms = list(lmatch.groups()) + [term1] + regex_terms[1:] + [term2] + list(rmatch.groups())
                    start, end = lmatch.start(), rmatch.end()
                else:  # this never happen
                    terms = regex_terms[1:]
                    start, end = match.start(), match.end()

                string = replace(string, start, end, sub_pat(ops.ops[op]['pat'], terms))
                match = ops.regex.search(string)

        return string