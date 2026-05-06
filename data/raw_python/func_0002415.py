def _do_perform_delete_on_model(self):
        """
        Perform the actual delete query on this model instance.
        """
        if self._force_deleting:
            return self.with_trashed().where(self.get_key_name(), self.get_key()).force_delete()

        return self._run_soft_delete()