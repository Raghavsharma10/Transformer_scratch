def unregister_signals(self):
        """Unregister signals."""
        from .models import Collection
        from .percolator import collection_inserted_percolator, \
            collection_removed_percolator, collection_updated_percolator
        # Unregister Record signals
        if hasattr(self, 'update_function'):
            signals.before_record_insert.disconnect(self.update_function)
            signals.before_record_update.disconnect(self.update_function)
        # Unregister collection signals
        if contains(Collection, 'after_insert',
                    collection_inserted_percolator):
            remove(Collection, 'after_insert', collection_inserted_percolator)
            remove(Collection, 'after_update', collection_updated_percolator)
            remove(Collection, 'after_delete', collection_removed_percolator)