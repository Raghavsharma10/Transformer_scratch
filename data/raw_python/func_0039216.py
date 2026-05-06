def bind(self,instance,setter,valueChangedSignal,getter = None):
        """
        Creates an endpoint and call bindToEndpoint(endpoint). This is a convenience method.

        Parameters:
            instance -- the object instance to which the getter, setter and changedSignal belong
            setter -- the value setter method
            valueChangedSignal -- the pyqtSignal that is emitted with the value changes
            getter -- the value getter method (default None)
                      If None, the signal argument(s) are passed to the setter method
        """

        endpoint = BindingEndpoint(instance,setter,valueChangedSignal,getter=getter)
        self.bindToEndPoint(endpoint)