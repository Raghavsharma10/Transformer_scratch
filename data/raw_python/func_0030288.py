def add_column(self, position, source_header, datatype, **kwargs):
        """
        Add a column to the source table.
        :param position: Integer position of the column started from 1.
        :param source_header: Name of the column, as it exists in the source file
        :param datatype: Python datatype ( str, int, float, None ) for the column
        :param kwargs:  Other source record args.
        :return:
        """
        from ..identity import GeneralNumber2

        c = self.column(source_header)
        c_by_pos = self.column(position)

        datatype = 'str' if datatype == 'unicode' else datatype

        assert not c or not c_by_pos or c.vid == c_by_pos.vid

        # Convert almost anything to True / False
        if 'has_codes' in kwargs:
            FALSE_VALUES = ['False', 'false', 'F', 'f', '', None, 0, '0']
            kwargs['has_codes'] = False if kwargs['has_codes'] in FALSE_VALUES else True

        if c:

            # Changing the position can result in conflicts
            assert not c_by_pos or c_by_pos.vid == c.vid

            c.update(
                position=position,
                datatype=datatype.__name__ if isinstance(datatype, type) else datatype,
                **kwargs)

        elif c_by_pos:

            # FIXME This feels wrong; there probably should not be any changes to the both
            # of the table, since then it won't represent the previouls source. Maybe all of the sources
            # should get their own tables initially, then affterward the duplicates can be removed.

            assert not c or c_by_pos.vid == c.vid

            c_by_pos.update(
                source_header=source_header,
                datatype=datatype.__name__ if isinstance(datatype, type) else datatype,
                **kwargs)

        else:

            assert not c and not c_by_pos

            # Hacking an id number, since I don't want to create a new Identity ObjectNUmber type
            c = SourceColumn(
                vid=str(GeneralNumber2('C', self.d_vid, self.sequence_id, int(position))),
                position=position,
                st_vid=self.vid,
                d_vid=self.d_vid,
                datatype=datatype.__name__ if isinstance(datatype, type) else datatype,
                source_header=source_header,
                **kwargs)

            self.columns.append(c)

        return c