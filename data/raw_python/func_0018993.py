def assignrepr(self, prefix: str = '') -> str:
        """Return a |repr| string with a prefixed assignment."""
        lines = ['%sNode("%s", variable="%s",'
                 % (prefix, self.name, self.variable)]
        if self.keywords:
            subprefix = '%skeywords=' % (' '*(len(prefix)+5))
            with objecttools.repr_.preserve_strings(True):
                with objecttools.assignrepr_tuple.always_bracketed(False):
                    line = objecttools.assignrepr_list(
                        sorted(self.keywords), subprefix, width=70)
            lines.append(line + ',')
        lines[-1] = lines[-1][:-1]+')'
        return '\n'.join(lines)