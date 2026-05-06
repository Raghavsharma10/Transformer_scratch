def variable_iter(self, base):
        """
        returns iterator over the cross product of the variables
        for this stanza
        """
        base_substs = dict(('<' + t + '>', u) for (t, u) in base.items())
        substs = []
        vals = []
        for with_defn in self.with_exprs:
            substs.append('<' + with_defn[0] + '>')
            vals.append(Host.expand_with(with_defn[1:]))
        for val_tpl in product(*vals):
            r = base_substs.copy()
            r.update(dict(zip(substs, val_tpl)))
            yield r