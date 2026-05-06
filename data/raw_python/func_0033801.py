def calibration(self):
    """
    Set the path to the calibration cache file for the given IFO.
    During S2 the Hanford 2km IFO had two calibration epochs, so
    if the start time is during S2, we use the correct cache file.
    """
    # figure out the name of the calibration cache files
    # as specified in the ini-file
    self.calibration_cache_path()

    if self.job().is_dax():
      # new code for DAX
      self.add_var_opt('glob-calibration-data','')
      cache_filename=self.get_calibration()
      pat = re.compile(r'(file://.*)')
      f = open(cache_filename, 'r')
      lines = f.readlines()

      # loop over entries in the cache-file...
      for line in lines:
        m = pat.search(line)
        if not m:
          raise IOError
        url = m.group(1)
        # ... and add files to input-file list
        path = urlparse.urlparse(url)[2]
        calibration_lfn = os.path.basename(path)
        self.add_input_file(calibration_lfn)
    else:
      # old .calibration for DAG's
      self.add_var_opt('calibration-cache', self.__calibration_cache)
      self.__calibration = self.__calibration_cache
      self.add_input_file(self.__calibration)