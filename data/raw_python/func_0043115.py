def register_signals(self):
        """Register signals."""
        from .models import Collection
        from .receivers import CollectionUpdater

        if self.app.config['COLLECTIONS_USE_PERCOLATOR']:
            from .percolator import collection_inserted_percolator, \
                collection_removed_percolator, \
                collection_updated_percolator
            # Register collection signals to update percolators
            listen(Collection, 'after_insert',
                   collection_inserted_percolator)
            listen(Collection, 'after_update',
                   collection_updated_percolator)
            listen(Collection, 'after_delete',
                   collection_removed_percolator)
        # Register Record signals to update record['_collections']
        self.update_function = CollectionUpdater(app=self.app)
        signals.before_record_insert.connect(self.update_function,
                                             weak=False)
        signals.before_record_update.connect(self.update_function,
                                             weak=False)