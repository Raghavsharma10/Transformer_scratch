def ac_viz(acdata):
  '''
  Adapted from Gerry Harp at SETI.
  
  Slightly massages the autocorrelated calculation result for better visualization.

  In particular, the natural log of the data are calculated and the
  values along the subband edges are set to the maximum value of the data, 
  and the t=0 delay of the autocorrelation result are set to the value of the t=-1 delay.

  This is allowed because the t=0, and subband edges do not carry any information. 

  To avoid log(0), a value of 0.000001 is added to all array elements before being logged. 
  '''

  acdata = np.log(acdata+0.000001)  # log to reduce darkening on sides of spectrum, due to AC triangling
  acdata[:, :, acdata.shape[2]/2] = acdata[:, :, acdata.shape[2]/2 - 1]  # vals at zero delay set to symmetric neighbor vals
  acdata[:, :, acdata.shape[2] - 1] = np.max(acdata)  # visualize subband edges

  return acdata