def add_aggregate(self, name, data_fac):
        """
        Add an aggregate target to this nest.


        Since nests added after the aggregate can access the construct returned
        by the factory function value, it can be mutated to provide additional
        values for use when the decorated function is called.

        To do something with the aggregates, you must :meth:`SConsWrap.pop`
        nest levels created between addition of the aggregate and then can add
        any normal targets you would like which take advantage of the targets
        added to the data structure.

        :param name: Name for the target in the nest
        :param data_fac: a nullary factory function which will be called
            immediately for each of the current control dictionaries and stored
            in each dictionary with the given name as in
            :meth:`SConsWrap.add_target`.
        """
        @self.add_target(name)
        def wrap(outdir, c):
            return data_fac()