def _reset(self) -> None:
        """Reset some of the state in the class for multi-searches."""
        self.project: str = namesgenerator.get_random_name()
        self._processed: List = list()
        self.results: List = list()