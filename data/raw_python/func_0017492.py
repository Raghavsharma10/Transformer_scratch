def Model(self):
        """Bind model to self database."""
        Model_ = self.app.config['PEEWEE_MODELS_CLASS']
        meta_params = {'database': self.database}
        if self.slaves and self.app.config['PEEWEE_USE_READ_SLAVES']:
            meta_params['read_slaves'] = self.slaves

        Meta = type('Meta', (), meta_params)
        return type('Model', (Model_,), {'Meta': Meta})