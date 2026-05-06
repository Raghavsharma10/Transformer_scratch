def load_file(self, fname, table=None, sep="\t", bins=False, indexes=None):
        """
        use some of the machinery in pandas to load a file into a table

        Parameters
        ----------

        fname : str
            filename or filehandle to load

        table : str
            table to load the file to

        sep : str
            CSV separator

        bins : bool
            add a "bin" column for efficient spatial queries.

        indexes : list[str]
            list of columns to index

        """
        convs = {"#chr": "chrom", "start": "txStart", "end": "txEnd", "chr":
                "chrom", "pos": "start", "POS": "start", "chromStart": "txStart",
                "chromEnd": "txEnd"}
        if table is None:
            import os.path as op
            table = op.basename(op.splitext(fname)[0]).replace(".", "_")
            print("writing to:", table, file=sys.stderr)

        from pandas.io import sql
        import pandas as pa
        from toolshed import nopen

        needs_name = False
        for i, chunk in enumerate(pa.read_csv(nopen(fname), iterator=True,
            chunksize=100000, sep=sep, encoding="latin-1")):
            chunk.columns = [convs.get(k, k) for k in chunk.columns]
            if not "name" in chunk.columns:
                needs_name = True
                chunk['name'] = chunk.get('chrom', chunk[chunk.columns[0]])
            if bins:
                chunk['bin'] = 1
            if i == 0 and not table in self.tables:
                flavor = self.url.split(":")[0]
                schema = sql.get_schema(chunk, table, flavor)
                print(schema)
                self.engine.execute(schema)
            elif i == 0:
                print >>sys.stderr,\
                        """adding to existing table, you may want to drop first"""

            tbl = getattr(self, table)._table
            cols = chunk.columns
            data = list(dict(zip(cols, x)) for x in chunk.values)
            if needs_name:
                for d in data:
                    d['name'] = "%s:%s" % (d.get("chrom"), d.get("txStart", d.get("chromStart")))
            if bins:
                for d in data:
                    d['bin'] = max(Genome.bins(int(d["txStart"]), int(d["txEnd"])))
            self.engine.execute(tbl.insert(), data)
            self.session.commit()
            if i > 0:
                print >>sys.stderr, "writing row:", i * 100000
        if "txStart" in chunk.columns:
            if "chrom" in chunk.columns:
                ssql = """CREATE INDEX "%s.chrom_txStart" ON "%s" (chrom, txStart)""" % (table, table)
            else:
                ssql = """CREATE INDEX "%s.txStart" ON "%s" (txStart)""" % (table, table)

            self.engine.execute(ssql)
        for index in (indexes or []):
            ssql = """CREATE INDEX "%s.%s" ON "%s" (%s)""" % (table,
                                index, table, index)
            self.engine.execute(ssql)

        if bins:
            ssql = """CREATE INDEX "%s.chrom_bin" ON "%s" (chrom, bin)""" % (table, table)
            self.engine.execute(ssql)

        self.session.commit()