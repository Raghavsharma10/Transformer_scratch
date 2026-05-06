def objects_to_record(self):
        """Write from object metadata to the record. Note that we don't write everything"""

        o = self.get_object()

        o.about = self._bundle.metadata.about
        o.identity = self._dataset.identity.ident_dict
        o.names = self._dataset.identity.names_dict
        o.contacts = self._bundle.metadata.contacts

        self.set_object(o)