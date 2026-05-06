def write(self, outfile=None, section=None):
        """
        Write the current ConfigObj as a file

        tekNico: FIXME: use StringIO instead of real files

        >>> filename = a.filename  # doctest: +SKIP
        >>> a.filename = 'test.ini'   # doctest: +SKIP
        >>> a.write()   # doctest: +SKIP
        >>> a.filename = filename   # doctest: +SKIP
        >>> a == ConfigObj('test.ini', raise_errors=True)   # doctest: +SKIP
        1
        >>> import os   # doctest: +SKIP
        >>> os.remove('test.ini')   # doctest: +SKIP
        """
        if self.indent_type is None:
            # this can be true if initialised from a dictionary
            self.indent_type = DEFAULT_INDENT_TYPE

        out = []
        cs = self._a_to_u('#')
        csp = self._a_to_u('# ')
        if section is None:
            int_val = self.interpolation
            self.interpolation = False
            section = self
            for line in self.initial_comment:
                line = self._decode_element(line)
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith(cs):
                    line = csp + line
                out.append(line)

        indent_string = self.indent_type * section.depth
        for entry in (section.scalars + section.sections):
            if entry in section.defaults:
                # don't write out default values
                continue
            for comment_line in section.comments[entry]:
                comment_line = self._decode_element(comment_line.lstrip())
                if comment_line and not comment_line.startswith(cs):
                    comment_line = csp + comment_line
                out.append(indent_string + comment_line)
            this_entry = section[entry]
            comment = self._handle_comment(section.inline_comments[entry])

            if isinstance(this_entry, dict):
                # a section
                out.append(self._write_marker(
                    indent_string,
                    this_entry.depth,
                    entry,
                    comment))
                out.extend(self.write(section=this_entry))
            else:
                out.append(self._write_line(
                    indent_string,
                    entry,
                    this_entry,
                    comment))

        if section is self:
            for line in self.final_comment:
                line = self._decode_element(line)
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith(cs):
                    line = csp + line
                out.append(line)
            self.interpolation = int_val

        if section is not self:
            return out

        if (self.filename is None) and (outfile is None):
            # output a list of lines
            # might need to encode
            # NOTE: This will *screw* UTF16, each line will start with the BOM
            if self.encoding:
                out = [l.encode(self.encoding) for l in out]
            if (self.BOM and ((self.encoding is None) or
                (BOM_LIST.get(self.encoding.lower()) == 'utf_8'))):
                # Add the UTF8 BOM
                if not out:
                    out.append('')
                out[0] = BOM_UTF8 + out[0]
            return out

        # Turn the list to a string, joined with correct newlines
        newline = self.newlines or os.linesep
        if (getattr(outfile, 'mode', None) is not None and outfile.mode == 'w'
            and sys.platform == 'win32' and newline == '\r\n'):
            # Windows specific hack to avoid writing '\r\r\n'
            newline = '\n'
        output = self._a_to_u(newline).join(out)
        if self.encoding:
            output = output.encode(self.encoding)
        if self.BOM and ((self.encoding is None) or match_utf8(self.encoding)):
            # Add the UTF8 BOM
            output = BOM_UTF8 + output

        if not output.endswith(newline):
            output += newline
        if outfile is not None:
            outfile.write(output)
        else:
            # !!! write mode was 'wb' but that fails in PY3K and we dont need
            h = open(self.filename, 'w')
            h.write(output)
            h.close()