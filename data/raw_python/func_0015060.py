def removeObserver(self, observer):
        """
        Removes an observer.

        @param observer: Previously added observer
        """
        with self._lock:
            self._weak_observers.remove(weakref.ref(observer))
            if len(self._weak_observers) == 0:
                self.stopThread()