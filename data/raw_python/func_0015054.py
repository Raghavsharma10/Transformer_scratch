def remove_all_observers(self):
        """
        Removes all registered observers.
        """
        for weak_observer in self._weak_observers:
            observer = weak_observer()
            if observer:
                self.remove_observer(observer)