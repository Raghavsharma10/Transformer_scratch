def prepare(
            self, engine=None, mode=None, model=None, index=None, key=None,
            filter=None, projection=None, consistent=None, forward=None, parallel=None):
        """Validates the search parameters and builds the base request dict for each Query/Scan call."""

        self.prepare_iterator_cls(engine, mode)
        self.prepare_model(model, index, consistent)
        self.prepare_key(key)
        self.prepare_projection(projection)
        self.prepare_filter(filter)
        self.prepare_constraints(forward, parallel)

        self.prepare_request()