def sort(self, name, start=None, num=None, by=None, get=None,
             desc=False, alpha=False, store=None, groups=False):
        """
        Sort and return the list, set or sorted set at ``name``.

        ``start`` and ``num`` allow for paging through the sorted data

        ``by`` allows using an external key to weight and sort the items.
            Use an "*" to indicate where in the key the item value is located

        ``get`` allows for returning items from external keys rather than the
            sorted data itself.  Use an "*" to indicate where int he key
            the item value is located

        ``desc`` allows for reversing the sort

        ``alpha`` allows for sorting lexicographically rather than numerically

        ``store`` allows for storing the result of the sort into
            the key ``store``

        ``groups`` if set to True and if ``get`` contains at least two
            elements, sort will return a list of tuples, each containing the
            values fetched from the arguments to ``get``.

        """
        with self.pipe as pipe:
            res = pipe.sort(self.redis_key(name), start=start, num=num,
                            by=by, get=get, desc=desc, alpha=alpha,
                            store=store, groups=groups)
            if store:
                return res
            f = Future()

            def cb():
                decode = self.valueparse.decode
                f.set([decode(v) for v in res.result])

            pipe.on_execute(cb)
            return f