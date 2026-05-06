def verify(self) -> None:
        """Raise a |RuntimeError| if the network's shape is not defined
        completely.

        >>> from hydpy import ANN
        >>> ANN(None).verify()
        Traceback (most recent call last):
        ...
        RuntimeError: The shape of the the artificial neural network \
parameter `ann` of element `?` has not been defined so far.
        """
        if not self.__protectedproperties.allready(self):
            raise RuntimeError(
                'The shape of the the artificial neural network '
                'parameter %s has not been defined so far.'
                % objecttools.elementphrase(self))