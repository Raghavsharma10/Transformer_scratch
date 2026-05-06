def _get_meta_options(self) -> List[MetaOption]:
        """
        Returns a list of :class:`MetaOption` instances that this factory supports.
        """
        return [option if isinstance(option, MetaOption) else option()
                for option in self._options]