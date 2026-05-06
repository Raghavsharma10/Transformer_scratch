def assignrepr(self, prefix: str) -> str:
        """Return a |repr| string with a prefixed assignment."""
        with objecttools.repr_.preserve_strings(True):
            with objecttools.assignrepr_tuple.always_bracketed(False):
                blanks = ' ' * (len(prefix) + 8)
                lines = ['%sElement("%s",' % (prefix, self.name)]
                for groupname in ('inlets', 'outlets', 'receivers', 'senders'):
                    group = getattr(self, groupname, Node)
                    if group:
                        subprefix = '%s%s=' % (blanks, groupname)
                        # pylint: disable=not-an-iterable
                        # because pylint is wrong
                        nodes = [str(node) for node in group]
                        # pylint: enable=not-an-iterable
                        line = objecttools.assignrepr_list(
                            nodes, subprefix, width=70)
                        lines.append(line + ',')
                if self.keywords:
                    subprefix = '%skeywords=' % blanks
                    line = objecttools.assignrepr_list(
                        sorted(self.keywords), subprefix, width=70)
                    lines.append(line + ',')
                lines[-1] = lines[-1][:-1]+')'
                return '\n'.join(lines)