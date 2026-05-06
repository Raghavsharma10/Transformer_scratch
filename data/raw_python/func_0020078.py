def GetNeighbors(ID, model = None, neighbors = None, mag_range = None, 
                 cdpp_range = None, aperture_name = None, 
                 cadence = 'lc', **kwargs):
  '''
  Return `neighbors` random bright stars on the same module as `EPIC`.
  
  :param int ID: The target ID number
  :param str model: The :py:obj:`everest` model name. Only used when imposing CDPP bounds. Default :py:obj:`None`
  :param int neighbors: Number of neighbors to return. Default None
  :param str aperture_name: The name of the aperture to use. Select `custom` to call \
         :py:func:`GetCustomAperture`. Default :py:obj:`None`
  :param str cadence: The light curve cadence. Default `lc`
  :param tuple mag_range: (`low`, `high`) values for the Kepler magnitude. Default :py:obj:`None`
  :param tuple cdpp_range: (`low`, `high`) values for the de-trended CDPP. Default :py:obj:`None`
  
  '''
  
  raise NotImplementedError('This mission is not yet supported.')