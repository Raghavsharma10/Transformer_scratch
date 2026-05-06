def __set_output(self):
    """
    Private method to set the file to write the cache to. Automaticaly set
    once the ifo, start and end times have been set.
    """
    if self.__start and self.__end and self.__observatory and self.__type:
      self.__output = os.path.join(self.__job.get_cache_dir(), self.__observatory + '-' + self.__type +'_CACHE' + '-' + str(self.__start) + '-' + str(self.__end - self.__start) + '.lcf')
      self.set_output(self.__output)