def add(self, observableElement):
        """
        add an observable element

        :param str observableElement: the name of the observable element
        :raises RuntimeError: if element name already exist in the store
        """
        if observableElement not in self._observables:
            self._observables.append(observableElement)
        else:
            raise RuntimeError(
                "{0} is already an observable element"
                .format(observableElement))