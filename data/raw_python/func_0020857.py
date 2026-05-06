def unregister_signals(self):
        """Unregister signals."""
        # Unregister Record signals
        if hasattr(self, 'update_function'):
            records_signals.before_record_insert.disconnect(
                self.update_function)
            records_signals.before_record_update.disconnect(
                self.update_function)
        self.unregister_signals_oaiset()