def get_bool_attr(self, name):
    """ Returns the value of a boolean HTML attribute like `checked` or `disabled`
    """
    val = self.get_attr(name)
    return val is not None and val.lower() in ("true", name)