def GetData(ID, season = None, cadence = 'lc', clobber = False, delete_raw = False, 
            aperture_name = None, saturated_aperture_name = None,
            max_pixels = None, download_only = False, saturation_tolerance = None, 
            bad_bits = None, **kwargs):
  '''
  Returns a :py:obj:`DataContainer` instance with the raw data for the target.
  
  :param int ID: The target ID number
  :param int season: The observing season. Default :py:obj:`None`
  :param str cadence: The light curve cadence. Default `lc`
  :param bool clobber: Overwrite existing files? Default :py:obj:`False`
  :param bool delete_raw: Delete the FITS TPF after processing it? Default :py:obj:`False`
  :param str aperture_name: The name of the aperture to use. Select `custom` to call \
         :py:func:`GetCustomAperture`. Default :py:obj:`None`
  :param str saturated_aperture_name: The name of the aperture to use if the target is \
         saturated. Default :py:obj:`None`
  :param int max_pixels: Maximum number of pixels in the TPF. Default :py:obj:`None`
  :param bool download_only: Download raw TPF and return? Default :py:obj:`False`
  :param float saturation_tolerance: Target is considered saturated if flux is within \
         this fraction of the pixel well depth. Default :py:obj:`None`
  :param array_like bad_bits: Flagged :py:obj`QUALITY` bits to consider outliers when \
         computing the model. Default :py:obj:`None`
  
  '''
  
  raise NotImplementedError('This mission is not yet supported.')