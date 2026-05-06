def bindToProperty(self,instance,propertyName,useGetter=False):
        """
        2-way binds to an instance property.

        Parameters:
        - instance -- the object instance
        - propertyName -- the name of the property to bind to
        - useGetter: when True, calls the getter method to obtain the value. When False, the signal argument is used as input for the target setter. (default False)

        Notes:
        2-way binds to an instance property according to one of the following naming conventions:

        @property, propertyName.setter and pyqtSignal
        - getter: propertyName
        - setter: propertyName
        - changedSignal: propertyNameChanged

        getter, setter and pyqtSignal (this is used when binding to standard QWidgets like QSpinBox)
        - getter: propertyName()
        - setter: setPropertyName()
        - changedSignal: propertyNameChanged
        """
        endpoint = BindingEndpoint.forProperty(instance,propertyName,useGetter = useGetter)

        self.bindToEndPoint(endpoint)