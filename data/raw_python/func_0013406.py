def _add_to_typedef(self, typedef_curr, line, lnum):
        """Add new fields to the current typedef."""
        mtch = re.match(r'^(\S+):\s*(\S.*)$', line)
        if mtch:
            field_name = mtch.group(1)
            field_value = mtch.group(2).split('!')[0].rstrip()

            if field_name == "id":
                self._chk_none(typedef_curr.id, lnum)
                typedef_curr.id = field_value
            elif field_name == "name":
                self._chk_none(typedef_curr.name, lnum)
                typedef_curr.name = field_value
            elif field_name == "transitive_over":
                typedef_curr.transitive_over.append(field_value)
            elif field_name == "inverse_of":
                self._chk_none(typedef_curr.inverse_of, lnum)
                typedef_curr.inverse_of = field_value
            # Note: there are other tags that aren't imported here.
        else:
            self._die("UNEXPECTED FIELD CONTENT: {L}\n".format(L=line), lnum)