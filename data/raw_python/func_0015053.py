def add_observer(self, observer):
        """
        Adds weak ref to an observer.

        @param observer: Instance of class registered with PowerManagementObserver
        @raise TypeError: If observer is not registered with PowerManagementObserver abstract class
        """
        if not isinstance(observer, PowerManagementObserver):
            raise TypeError("observer MUST conform to power.PowerManagementObserver")
        self._weak_observers.append(weakref.ref(observer))