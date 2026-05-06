def process_op(self, ns, raw):
        """ Processes a single operation from the oplog.

        Performs a switch by raw['op']:
            "i" insert
            "u" update
            "d" delete
            "c" db cmd
            "db" declares presence of a database
            "n" no op
        """
        # Compute the document id of the document that will be altered
        # (in case of insert, update or delete).
        docid = self.__get_id(raw)

        op = raw['op']
        if op == 'i':
            self.insert(ns=ns, docid=docid, raw=raw)
        elif op == 'u':
            self.update(ns=ns, docid=docid, raw=raw)
        elif op == 'd':
            self.delete(ns=ns, docid=docid, raw=raw)
        elif op == 'c':
            self.command(ns=ns, raw=raw)
        elif op == 'db':
            self.db_declare(ns=ns, raw=raw)
        elif op == 'n':
            self.noop()
        else:
            logging.error("Unknown op: %r" % op)

        # Save timestamp of last processed oplog.
        self.ts = raw['ts']