def existingInStore(cls, store, storeID, attrs):
        """Create and return a new instance from a row from the store."""
        self = cls.__new__(cls)

        self.__justCreated = False
        self.__subinit__(__store=store,
                         storeID=storeID,
                         __everInserted=True)

        schema = self.getSchema()
        assert len(schema) == len(attrs), "invalid number of attributes"
        for data, (name, attr) in zip(attrs, schema):
            attr.loaded(self, data)
        self.activate()
        return self