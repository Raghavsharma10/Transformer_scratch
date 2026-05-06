def register(self, *args):
        """Register a configurable component in the configuration schema
        store"""

        super(ConfigurableMeta, self).register(*args)
        from hfos.database import configschemastore
        # self.log('ADDING SCHEMA:')
        # pprint(self.configschema)
        configschemastore[self.name] = self.configschema