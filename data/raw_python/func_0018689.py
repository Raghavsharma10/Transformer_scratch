def run_restore(self, snapshot: Dict[Union[str, Key], Any]) -> 'BaseItemCollection':
        """
        Restores the state of a collection from a snapshot
        """
        try:

            for name, snap in snapshot.items():
                if isinstance(name, Key):
                    self._nested_items[name.group].run_restore(snap)
                else:
                    self._nested_items[name].run_restore(snap)
            return self

        except Exception as e:
            raise SnapshotError('Error while restoring snapshot: {}'.format(self._snapshot)) from e