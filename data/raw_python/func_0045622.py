def compamp_to_spectrogram(compamp):
  '''
  Returns spectrogram, with each row containing the measured power spectrum for a XX second time sample.

  Using this function is shorthand for:
      aca = ibmseti.compamp.Compamp(raw_data)
      power = ibmseti.dsp.complex_to_power(aca.complex_data(), aca.header()['over_sampling'])
      spectrogram = ibmseti.dsp.reshape_to_2d(power)

  Example Usage: 
      import ibmseti
      import matplotlib.pyplot as plt
      plt.ion()
  
      aca = ibmseti.compamp.Compamp(raw_data)

      spectrogram = ibmseti.dsp.compamp_to_spectrogram(aca)
      time_bins = ibmseti.dsp.time_bins( aca.header() )
      freq_bins = ibmseti.dsp.frequency_bins( aca.header() )

      fig, ax = plt.subplots()
      ax.pcolormesh(freq_bins, time_bins, spectrogram)

      #Time is on the horizontal axis and frequency is along the vertical.

  '''

  power = complex_to_power(compamp.complex_data(), compamp.header()['over_sampling'])
  
  return reshape_to_2d(power)