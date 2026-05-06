def addObserver(self, observer):
        """
        Adds weak ref to an observer.

        @param observer: Instance of class that implements on_power_source_notification()
        """
        with self._lock:
            self._weak_observers.append(weakref.ref(observer))
            if len(self._weak_observers) == 1:
                self.startThread()