def bound_variable(self):
        """Return an object with a value property and a changed_event.

        The value property returns the value of the variable. The changed_event is fired
        whenever the value changes.
        """
        class BoundVariable:
            def __init__(self, variable):
                self.__variable = variable
                self.changed_event = Event.Event()
                self.needs_rebind_event = Event.Event()
                def property_changed(key):
                    if key == "value":
                        self.changed_event.fire()
                self.__variable_property_changed_listener = variable.property_changed_event.listen(property_changed)
            @property
            def value(self):
                return self.__variable.value
            def close(self):
                self.__variable_property_changed_listener.close()
                self.__variable_property_changed_listener = None

        return BoundVariable(self)