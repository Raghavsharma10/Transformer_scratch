def set_observatory(self,obs):
    """
    Set the IFO to retrieve data for. Since the data from both Hanford
    interferometers is stored in the same frame file, this takes the first
    letter of the IFO (e.g. L or H) and passes it to the --observatory option
    of LSCdataFind.
    @param obs: IFO to obtain data for.
    """
    self.add_var_opt('observatory',obs)
    self.__observatory = str(obs)
    self.__set_output()