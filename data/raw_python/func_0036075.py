def drop_index(self, raw):
        """ Executes a drop index command.

            { "op" : "c",
              "ns" : "testdb.$cmd",
              "o" : { "dropIndexes" : "testcoll",
            		  "index" : "nuie_1" } }
        """
        dbname = raw['ns'].split('.', 1)[0]
        collname = raw['o']['dropIndexes']
        self.dest[dbname][collname].drop_index(raw['o']['index'])