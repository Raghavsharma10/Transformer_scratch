def set_attr(self, name, value):
    """ Sets the value of an attribute. """
    self.exec_script("node.setAttribute(%s, %s)" % (repr(name), repr(value)))