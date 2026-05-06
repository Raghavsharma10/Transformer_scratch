def set(self, x):
        """
        Set variable values via a dictionary mapping name to value.
        """
        for name, value in iter(x.items()):
            if hasattr(value, "ndim"):
                if self[name].value.ndim < value.ndim:
                    self[name].value.itemset(value.squeeze())
                else:
                    self[name].value = value
            else:
                self[name].value.itemset(value)