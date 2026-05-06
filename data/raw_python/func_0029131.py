def update_source_hashes(self, instance):
        """
        Stores hashes of the source image files so that they can be compared
        later to see whether the source image has changed (and therefore whether
        the spec file needs to be regenerated).

        """
        self.init_instance(instance)
        instance._ik['source_hashes'] = dict(
            (attname, hash(getattr(instance, attname)))
            for attname in self.get_source_fields(instance))
        return instance._ik['source_hashes']