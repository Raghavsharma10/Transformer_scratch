def forProperty(instance,propertyName,useGetter=False):
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
        assert isinstance(propertyName,str)

        if propertyName.startswith("get") or propertyName.startswith("set"):
            #property is a getter function or a setter function, assume a corresponding setter/getter function exists
            getterName = "get" + propertyName[3:]
            setterName = "set" + propertyName[3:]
            if len(propertyName[3:]) > 1:
                signalName = propertyName[3].lower() + propertyName[4:] + "Changed"
            else:
                signalName = propertyName.lower() + "Changed"

            assert hasattr(instance,getterName)
            assert hasattr(instance,setterName)
            assert hasattr(instance,signalName)
            getter = getattr(instance,getterName)
            setter = getattr(instance,setterName)
            signal = getattr(instance,signalName)

        elif hasattr(instance, propertyName) and callable(getattr(instance,propertyName)):
            #property is a getter function without the "get" prefix. Assume a corresponding setter method exists
            getterName = propertyName
            setterName = "set" + propertyName.capitalize()
            signalName = propertyName + "Changed"

            assert hasattr(instance,setterName)
            assert hasattr(instance,signalName)
            getter = getattr(instance,getterName)
            setter = getattr(instance,setterName)
            signal = getattr(instance,signalName)


        elif hasattr(instance, propertyName):
            #property is real property. Assume it is not readonly
            signalName = propertyName + "Changed"
            assert hasattr(instance,signalName)

            getter = lambda: getattr(instance,propertyName)
            setter = lambda value: setattr(instance,propertyName,value)
            signal = getattr(instance,signalName)

        else:
            #property is a virtual property. There should be getPropertyName and setPropertyName methods
            if len(propertyName) > 1:
                getterName = "get" + propertyName[0].upper() + propertyName[1:]
                setterName = "set" + propertyName[0].upper() + propertyName[1:]
                signalName = propertyName + "Changed"
            else:
                getterName = "get" + propertyName.upper()
                setterName = "set" + propertyName.upper()
                signalName = propertyName.lower() + "Changed"

            assert hasattr(instance,getterName)
            assert hasattr(instance,setterName)
            assert hasattr(instance,signalName)

            getter = getattr(instance,getterName)
            setter = getattr(instance,setterName)
            signal = getattr(instance,signalName)

        return BindingEndpoint(instance, setter, signal, getter = getter if useGetter else None)