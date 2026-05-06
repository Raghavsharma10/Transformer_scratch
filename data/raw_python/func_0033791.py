def get_config(self,sec,opt):
    """
    Get the configration variable in a particular section of this jobs ini
    file.
    @param sec: ini file section.
    @param opt: option from section sec.
    """
    return string.strip(self.__cp.get(sec,opt))