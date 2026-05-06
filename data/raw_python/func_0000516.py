def parse(self, data=None, table_name=None):
        """Parse the lines from index i

        :param data: optional, store the parsed result to it when specified
        :param table_name: when inside a table array, it is the table name
        """
        temp = self.dict_()
        sub_table = None
        is_array = False
        line = ''
        while True:
            line = self._readline()
            if not line:
                self._store_table(sub_table, temp, is_array, data=data)
                break       # EOF
            if BLANK_RE.match(line):
                continue
            if TABLE_RE.match(line):
                next_table = self.split_string(
                    TABLE_RE.match(line).group(1), '.', False)
                if table_name and not contains_list(next_table, table_name):
                    self._store_table(sub_table, temp, is_array, data=data)
                    break
                table = cut_list(next_table, table_name)
                if sub_table == table:
                    raise TomlDecodeError(self.lineno, 'Duplicate table name'
                                          'in origin: %r' % sub_table)
                else:       # different table name
                    self._store_table(sub_table, temp, is_array, data=data)
                    sub_table = table
                    is_array = False
            elif TABLE_ARRAY_RE.match(line):
                next_table = self.split_string(
                    TABLE_ARRAY_RE.match(line).group(1), '.', False)
                if table_name and not contains_list(next_table, table_name):
                    # Out of current loop
                    # write current data dict to table dict
                    self._store_table(sub_table, temp, is_array, data=data)
                    break
                table = cut_list(next_table, table_name)
                if sub_table == table and not is_array:
                    raise TomlDecodeError(self.lineno, 'Duplicate name of '
                                          'table and array of table: %r'
                                          % sub_table)
                else:   # Begin a nested loop
                    # Write any temp data to table dict
                    self._store_table(sub_table, temp, is_array, data=data)
                    sub_table = table
                    is_array = True
                    self.parse(temp, next_table)
            elif KEY_RE.match(line):
                m = KEY_RE.match(line)
                keys = self.split_string(m.group(1), '.')
                value = self.converter.convert(line[m.end():])
                if value is None:
                    raise TomlDecodeError(self.lineno, 'Value is missing')
                self._store_table(keys[:-1], {keys[-1]: value}, data=temp)
            else:
                raise TomlDecodeError(self.lineno,
                                      'Pattern is not recognized: %r' % line)
        # Rollback to the last line for next parse
        # This will do nothing if EOF is hit
        self.instream.seek(self.instream.tell() - len(line))
        self.lineno -= 1