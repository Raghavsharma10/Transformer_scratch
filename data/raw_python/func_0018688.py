def _snapshot(self) -> Dict[str, Any]:
        """
        Implements snapshot for collections by recursively invoking snapshot of all child items
        """
        try:
            return {name: item._snapshot for name, item in self._nested_items.items()}
        except Exception as e:
            raise SnapshotError('Error while creating snapshot for {}'.format(self._name)) from e