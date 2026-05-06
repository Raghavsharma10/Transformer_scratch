def remove(self, observableElement):
        """
        remove an obsrvable element

        :param str observableElement: the name of the observable element
        """
        if observableElement in self._observables:
            self._observables.remove(observableElement)