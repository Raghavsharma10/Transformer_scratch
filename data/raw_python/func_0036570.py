def flowtable(self):
        """ get a flat flow table globally
        """
        ftable = dict()
        for table in self.flow_table:
            for k, v in table.items():
                if k not in ftable:
                    ftable[k] = set(v)
                else:
                    [ftable[k].add(i) for i in v]
        # convert set to list
        for k in ftable:
            ftable[k] = list(ftable[k])
        return ftable