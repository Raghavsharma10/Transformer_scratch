def table(self, oid, columns=None, column_value_mapping=None, non_repeaters=0,
              max_repetitions=20, fetch_all_columns=True):
        """
        Get a table of values with the given OID prefix.
        """
        snmpsecurity = self._get_snmp_security()
        base_oid = oid.strip(".")

        if not fetch_all_columns and not columns:
            raise ValueError("please use the columns argument to "
                             "indicate which columns to fetch")

        if fetch_all_columns:
            columns_to_fetch = [""]
        else:
            columns_to_fetch = ["." + str(col_id) for col_id in columns.keys()]

        full_obj_table = []

        for col in columns_to_fetch:
            try:
                engine_error, pdu_error, pdu_error_index, obj_table = self._cmdgen.bulkCmd(
                    snmpsecurity,
                    cmdgen.UdpTransportTarget((self.host, self.port), timeout=self.timeout,
                                              retries=self.retries),
                    non_repeaters,
                    max_repetitions,
                    oid + col,
                )

            except Exception as e:
                raise SNMPError(e)
            if engine_error:
                raise SNMPError(engine_error)
            if pdu_error:
                raise SNMPError(pdu_error.prettyPrint())

            # remove any trailing rows from the next subtree
            try:
                while not str(obj_table[-1][0][0].getOid()).lstrip(".").startswith(
                    base_oid + col + "."
                ):
                    obj_table.pop()
            except IndexError:
                pass

            # append this column to full result
            full_obj_table += obj_table

        t = Table(columns=columns, column_value_mapping=column_value_mapping)

        for row in full_obj_table:
            for name, value in row:
                oid = str(name.getOid()).strip(".")
                value = _convert_value_to_native(value)
                column, row_id = oid[len(base_oid) + 1:].split(".", 1)
                t._add_value(int(column), row_id, value)

        return t