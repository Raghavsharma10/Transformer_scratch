def unregister_signals_oaiset(self):
        """Unregister signals oaiset."""
        from .models import OAISet
        from .receivers import after_insert_oai_set, \
            after_update_oai_set, after_delete_oai_set
        if contains(OAISet, 'after_insert', after_insert_oai_set):
            remove(OAISet, 'after_insert', after_insert_oai_set)
            remove(OAISet, 'after_update', after_update_oai_set)
            remove(OAISet, 'after_delete', after_delete_oai_set)