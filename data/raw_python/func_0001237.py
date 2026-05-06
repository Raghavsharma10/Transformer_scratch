def summarize(self, rows):
        """Return summary rows for `rows`.

        Parameters
        ----------
        rows : list of dicts
            Normalized rows to summarize.

        Returns
        -------
        A list of summary rows.  Each row is a tuple where the first item is
        the data and the second is a dict of keyword arguments that can be
        passed to StyleFields.render.
        """
        columns = list(rows[0].keys())
        agg_styles = {c: self.style[c]["aggregate"]
                      for c in columns if "aggregate" in self.style[c]}

        summaries = {}
        for col, agg_fn in agg_styles.items():
            lgr.debug("Summarizing column %r with %r", col, agg_fn)
            colvals = filter(lambda x: not isinstance(x, Nothing),
                             (row[col] for row in rows))
            summaries[col] = agg_fn(list(colvals))

        # The rest is just restructuring the summaries into rows that are
        # compatible with pyout.Content.  Most the complexity below comes from
        # the fact that a summary function is allowed to return either a single
        # item or a list of items.
        maxlen = max(len(v) if isinstance(v, list) else 1
                     for v in summaries.values())
        summary_rows = []
        for rowidx in range(maxlen):
            sumrow = {}
            for column, values in summaries.items():
                if isinstance(values, list):
                    if rowidx >= len(values):
                        continue
                    sumrow[column] = values[rowidx]
                elif rowidx == 0:
                    sumrow[column] = values

            for column in columns:
                if column not in sumrow:
                    sumrow[column] = ""

            summary_rows.append((sumrow,
                                 {"style": self.style.get("aggregate_"),
                                  "adopt": False}))
        return summary_rows