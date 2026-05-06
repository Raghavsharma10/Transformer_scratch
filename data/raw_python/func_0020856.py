def register_signals_oaiset(self):
        """Register OAISet signals to update records."""
        from .models import OAISet
        from .receivers import after_insert_oai_set, \
            after_update_oai_set, after_delete_oai_set
        listen(OAISet, 'after_insert', after_insert_oai_set)
        listen(OAISet, 'after_update', after_update_oai_set)
        listen(OAISet, 'after_delete', after_delete_oai_set)