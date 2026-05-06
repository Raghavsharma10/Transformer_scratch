def _add_or_update_records(cls, conn: Connection, table: Table,
                               records: List["I2B2CoreWithUploadId"]) -> Tuple[int, int]:
        """Add or update the supplied table as needed to reflect the contents of records

        :param table: i2b2 sql connection
        :param records: records to apply
        :return: number of records added / modified
        """
        num_updates = 0
        num_inserts = 0
        inserts = []
        # Iterate over the records doing updates
        # Note: This is slow as molasses - definitely not optimal for batch work, but hopefully we'll be dealing with
        #    thousands to tens of thousands of records.  May want to move to ORM model if this gets to be an issue
        for record in records:
            keys = [(table.c[k] == getattr(record, k)) for k in cls.key_fields]
            key_filter = I2B2CoreWithUploadId._nested_fcn(and_, keys)
            rec_exists = conn.execute(select([table.c.upload_id]).where(key_filter)).rowcount
            if rec_exists:
                known_values = {k: v for k, v in as_dict(record).items()
                                if v is not None and k not in cls._no_update_fields and
                                k not in cls.key_fields}
                vals = [table.c[k] != v for k, v in known_values.items()]
                val_filter = I2B2CoreWithUploadId._nested_fcn(or_, vals)
                known_values['update_date'] = record.update_date
                upd = update(table).where(and_(key_filter, val_filter)).values(known_values)
                num_updates += conn.execute(upd).rowcount
            else:
                inserts.append(as_dict(record))
        if inserts:
            if cls._check_dups:
                dups = cls._check_for_dups(inserts)
                nprints = 0
                if dups:
                    print("{} duplicate records encountered".format(len(dups)))
                    for k, vals in dups.items():
                        if len(vals) == 2 and vals[0] == vals[1]:
                            inserts.remove(vals[1])
                        else:
                            if nprints < 20:
                                print("Key: {} has a non-identical dup".format(k))
                            elif nprints == 20:
                                print(".... more ...")
                            nprints += 1
                            for v in vals[1:]:
                                inserts.remove(v)
            # TODO: refactor this to load on a per-resource basis.  Temporary fix
            for insert in ListChunker(inserts, 500):
                num_inserts += conn.execute(table.insert(), insert).rowcount
        return num_inserts, num_updates