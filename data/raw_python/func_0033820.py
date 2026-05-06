def set_end(self,time):
    """
    Set the end time of the datafind query.
    @param time: GPS end time of query.
    """
    self.add_var_opt('gps-end-time', time)
    self.__end = time
    self.__set_output()