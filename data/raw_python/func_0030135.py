def detail_dict(self):
        """A more detailed dict that includes the descriptions, sub descriptions, table
        and columns."""

        d = self.dict

        def aug_col(c):
            d = c.dict
            d['stats'] = [s.dict for s in c.stats]
            return d

        d['table'] = self.table.dict
        d['table']['columns'] = [aug_col(c) for c in self.table.columns]

        return d