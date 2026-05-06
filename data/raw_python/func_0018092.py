def fasper(x, y, ofac, hifac, n_threads, MACC=4):
    """
    Given abscissas x (which need not be equally spaced) and ordinates
    y, and given a desired oversampling factor ofac (a typical value
    being 4 or larger). this routine creates an array wk1 with a
    sequence of nout increasing frequencies (not angular frequencies)
    up to hifac times the "average" Nyquist frequency, and creates
    an array wk2 with the values of the Lomb normalized periodogram at
    those frequencies. The arrays x and y are not altered. This
    routine also returns jmax such that wk2(jmax) is the maximum
    element in wk2, and prob, an estimate of the significance of that
    maximum against the hypothesis of random noise. A small value of prob
    indicates that a significant periodic signal is present.
    
    Reference: 
        Press, W. H. & Rybicki, G. B. 1989
        ApJ vol. 338, p. 277-280.
        Fast algorithm for spectral analysis of unevenly sampled data
        (1989ApJ...338..277P)

    Arguments:
            X     : Abscissas array, (e.g. an array of times).
            Y     : Ordinates array, (e.g. corresponding counts).
            Ofac : Oversampling factor.
            Hifac : Hifac * "average" Nyquist frequency = highest frequency
                     for which values of the Lomb normalized periodogram will
                     be calculated.
            n_threads : number of threads to use.
            
     Returns:
            Wk1 : An array of Lomb periodogram frequencies.
            Wk2 : An array of corresponding values of the Lomb periodogram.
            Nout : Wk1 & Wk2 dimensions (number of calculated frequencies)
            Jmax : The array index corresponding to the MAX( Wk2 ).
            Prob : False Alarm Probability of the largest Periodogram value
            MACC : Number of interpolation points per 1/4 cycle
                        of highest frequency

    History:
        02/23/2009, v1.0, MF
            Translation of IDL code (orig. Numerical recipies)
    """
    #Check dimensions of input arrays
    n = long(len(x))
    if n != len(y):
        print('Incompatible arrays.')
        return
    
    #print x, y, hifac, ofac
    
    nout = int(0.5*ofac*hifac*n)
    nfreqt = long(ofac*hifac*n*MACC)     #Size the FFT as next power
    nfreq = 64                         # of 2 above nfreqt.

    while nfreq < nfreqt:
        nfreq = 2*nfreq

    ndim = long(2*nfreq)
    
    #Compute the mean, variance
    ave = y.mean()
    ##sample variance because the divisor is N-1
    var = ((y - y.mean())**2).sum()/(len(y) - 1) 
    # and range of the data.
    xmin = x.min()
    xmax = x.max()
    xdif = xmax - xmin

    #extrapolate the data into the workspaces
    if is_pyfftw:
        wk1 = pyfftw.n_byte_align_empty(int(ndim), 16, 'complex') * 0.
        wk2 = pyfftw.n_byte_align_empty(int(ndim), 16, 'complex') * 0.
    else:
        wk1 = zeros(ndim, dtype='complex')
        wk2 = zeros(ndim, dtype='complex')

    fac = ndim/(xdif*ofac)
    fndim = ndim
    ck = ((x - xmin)*fac) % fndim
    ckk = (2.0*ck) % fndim
    
    for j in range(0, n):
        __spread__(y[j] - ave, wk1, ndim, ck[j], MACC)
        __spread__(1.0, wk2, ndim, ckk[j], MACC)

    #Take the Fast Fourier Transforms.
    if is_pyfftw:
        fft_wk1 = pyfftw.builders.ifft(wk1, planner_effort='FFTW_ESTIMATE',
            threads=n_threads)
        wk1 = fft_wk1() * len(wk1)
        fft_wk2 = pyfftw.builders.ifft(wk2, planner_effort='FFTW_ESTIMATE',
            threads=n_threads)
        wk2 = fft_wk2() * len(wk2)
    else:
        wk1 = ifft(wk1)*len(wk1)
        wk2 = ifft(wk2)*len(wk1)

    wk1 = wk1[1:nout + 1]
    wk2 = wk2[1:nout + 1]
    rwk1 = wk1.real
    iwk1 = wk1.imag
    rwk2 = wk2.real
    iwk2 = wk2.imag
    
    df = 1.0/(xdif*ofac)
    
    #Compute the Lomb value for each frequency
    hypo2 = 2.0*abs(wk2)
    hc2wt = rwk2/hypo2
    hs2wt = iwk2/hypo2

    cwt = sqrt(0.5 + hc2wt)
    swt = sign(hs2wt)*(sqrt(0.5 - hc2wt))
    den = 0.5*n + hc2wt*rwk2 + hs2wt*iwk2
    cterm = (cwt*rwk1 + swt*iwk1)**2./den
    sterm = (cwt*iwk1 - swt*rwk1)**2./(n - den)

    wk1 = df*(arange(nout, dtype='float') + 1.)
    wk2 = (cterm + sterm)/(2.0*var)
    pmax = wk2.max()
    jmax = wk2.argmax()

    #Significance estimation
    #expy = exp(-wk2)                    
    #effm = 2.0*(nout)/ofac             
    #sig = effm*expy
    #ind = (sig > 0.01).nonzero()
    #sig[ind] = 1.0-(1.0-expy[ind])**effm

    #Estimate significance of largest peak value
    expy = exp(-pmax)                    
    effm = 2.0*(nout)/ofac
    prob = effm*expy

    if prob > 0.01: 
        prob = 1.0 - (1.0 - expy)**effm

    return wk1, wk2, nout, jmax, prob