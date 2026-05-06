def set_type(self,type):
    """
    sets the frame type that we are querying
    """
    self.add_var_opt('type',str(type))
    self.__type = str(type)
    self.__set_output()