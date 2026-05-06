def value(self):
    """ Returns the node's value. """
    if self.is_multi_select():
      return [opt.value()
              for opt in self.xpath(".//option")
              if opt["selected"]]
    else:
      return self._invoke("value")