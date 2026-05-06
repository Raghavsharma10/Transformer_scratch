def prepare(self, engine, mode, items) -> None:
        """
        Create a unique transaction id and dumps the items into a cached request object.
        """
        self.tx_id = str(uuid.uuid4()).replace("-", "")
        self.engine = engine
        self.mode = mode
        self.items = items
        self._prepare_request()