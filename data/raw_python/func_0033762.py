def set_grid_site(self,site):
    """
    Set the grid site to run on. If not specified,
    will not give hint to Pegasus
    """
    self.__grid_site=str(site)
    if site != 'local':
      self.set_executable_installed(False)