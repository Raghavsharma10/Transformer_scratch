def register_signals(self):
        """Register signals."""
        from .receivers import OAIServerUpdater
        # Register Record signals to update OAI informations
        self.update_function = OAIServerUpdater()
        records_signals.before_record_insert.connect(self.update_function,
                                                     weak=False)
        records_signals.before_record_update.connect(self.update_function,
                                                     weak=False)
        if self.app.config['OAISERVER_REGISTER_SET_SIGNALS']:
            self.register_signals_oaiset()