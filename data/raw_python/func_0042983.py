def _groupby(self, edges):
        """
        RETURNS LIST OF (coord, values) TUPLES, WHERE
            coord IS THE INDEX INTO self CUBE (-1 INDEX FOR COORDINATES NOT GROUPED BY)
            values ALL VALUES THAT BELONG TO THE SLICE

        """
        edges = FlatList([n for e in edges for n in _normalize_edge(e)])

        stacked = [e for e in self.edges if e.name in edges.name]
        remainder = [e for e in self.edges if e.name not in edges.name]
        selector = [1 if e.name in edges.name else 0 for e in self.edges]

        if len(stacked) + len(remainder) != len(self.edges):
            Log.error("can not find some edges to group by")
        # CACHE SOME RESULTS
        keys = edges.name
        getKey = [e.domain.getKey for e in self.edges]
        lookup = [[getKey[i](p) for p in e.domain.partitions+([None] if e.allowNulls else [])] for i, e in enumerate(self.edges)]

        if is_list(self.select):
            selects = listwrap(self.select)
            index, v = transpose(*self.data[selects[0].name].groupby(selector))

            coord = wrap([coord2term(c) for c in index])

            values = [v]
            for s in selects[1::]:
                i, v = transpose(*self.data[s.name].group_by(selector))
                values.append(v)

            output = transpose(coord, [Cube(self.select, remainder, {s.name: v[i] for i, s in enumerate(selects)}) for v in zip(*values)])
        elif not remainder:
            # v IS A VALUE, NO NEED TO WRAP IT IN A Cube
            output = (
                (
                    coord2term(coord),
                    v
                )
                for coord, v in self.data[self.select.name].groupby(selector)
            )
        else:
            output = (
                (
                    coord2term(coord),
                    Cube(self.select, remainder, v)
                )
                for coord, v in self.data[self.select.name].groupby(selector)
            )

        return output