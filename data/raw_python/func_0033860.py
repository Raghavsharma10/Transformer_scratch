def start_element(self, name, attrs):
    """
    Callback for start of an XML element. Checks to see if we are
    about to start a table that matches the ignore pattern.

    @param name: the name of the tag being opened
    @type name: string

    @param attrs: a dictionary of the attributes for the tag being opened
    @type attrs: dictionary
    """
    if name.lower() == "table":
      for attr in attrs.keys():
        if attr.lower() == "name":
          if self.__ignore_pat.search(attrs[attr]):
            self.__in_table = 1