def write(self, rows, keyed=False):
        """Write rows/keyed_rows to table
        """
        for row in rows:
            keyed_row = row
            if not keyed:
                keyed_row = dict(zip(self.__schema.field_names, row))
            keyed_row = self.__convert_row(keyed_row)
            if self.__check_existing(keyed_row):
                for wr in self.__insert():
                    yield wr
                ret = self.__update(keyed_row)
                if ret is not None:
                    yield WrittenRow(keyed_row, True, ret if self.__autoincrement else None)
                    continue
            self.__buffer.append(keyed_row)
            if len(self.__buffer) > BUFFER_SIZE:
                for wr in self.__insert():
                    yield wr
        for wr in self.__insert():
            yield wr