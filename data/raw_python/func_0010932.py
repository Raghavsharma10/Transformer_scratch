def emit(self, signal, value=None, gather=False):
        """Emits a signal, causing all slot methods connected with the signal to be called (optionally w/ related value)

           signal: the name of the signal to emit, must be defined in the classes 'signals' list.
           value: the value to pass to all connected slot methods.
           gather: if set, causes emit to return a list of all slot results
        """
        results = [] if gather else True
        if hasattr(self, 'connections') and signal in self.connections:
            for condition, values in self.connections[signal].items():
                if condition is None or condition == value or (callable(condition) and condition(value)):
                    for slot, transform in values.items():
                        if transform is not None:
                            if callable(transform):
                                used_value = transform(value)
                            elif isinstance(transform, str):
                                used_value = transform.format(value=value)
                            else:
                                used_value = transform
                        else:
                            used_value = value

                        if used_value is not None:
                            if(accept_arguments(slot, 1)):
                                result = slot(used_value)
                            elif(accept_arguments(slot, 0)):
                                result = slot()
                            else:
                                result = ''
                        else:
                            result = slot()

                        if gather:
                            results.append(result)

        return results