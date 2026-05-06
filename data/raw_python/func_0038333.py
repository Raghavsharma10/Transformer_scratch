def convert_spectrum(File, verbose=False):
    """
    Converts a SPECTRUM data type stored in the database into a (W,F,E) sequence of arrays.

    Parameters
    ----------
    File: str
        The URL or filepath of the file to be converted into arrays.
    verbose: bool
        Whether or not to display some diagnostic information (Default: False)

    Returns
    -------
    sequence
        The converted spectrum.

    """
    spectrum, header = '', ''
    if isinstance(File, type(b'')):  # Decode if needed (ie, for Python 3)
        File = File.decode('utf-8')

    if isinstance(File, (str, type(u''))):

        # Convert variable path to absolute path
        if File.startswith('$'):
            abspath = os.popen('echo {}'.format(File.split('/')[0])).read()[:-1]
            if abspath: File = File.replace(File.split('/')[0], abspath)

        if File.startswith('http'):
            if verbose: print('Downloading {}'.format(File))
            downloaded_file = download_file(File, cache=True)  # download only once
        else:
            downloaded_file = File

        try:  # Try FITS files first
            # Get the data
            # try:
            spectrum, header = pf.getdata(downloaded_file, cache=True, header=True)
            # except:
            #     spectrum, header = pf.getdata(File, cache=False, header=True)

            # Check the key type
            KEY_TYPE = ['CTYPE1']
            setType = set(KEY_TYPE).intersection(set(header.keys()))
            if len(setType) == 0:
                isLinear = True
            else:
                valType = header[setType.pop()]
                isLinear = valType.strip().upper() == 'LINEAR'

            # Get wl, flux & error data from fits file
            spectrum = __get_spec(spectrum, header, File)

            # Generate wl axis when needed
            if not isinstance(spectrum[0], np.ndarray):
                tempwav = __create_waxis(header, len(spectrum[1]), File)

                # Check to see if it's a FIRE spectrum with CDELT1, if so needs wlog=True
                if 'INSTRUME' in header.keys():
                    if header['INSTRUME'].strip() == 'FIRE' and 'CDELT1' in header.keys():
                        tempwav = __create_waxis(header, len(spectrum[1]), File, wlog=True)

                spectrum[0] = tempwav

            # If no wl axis generated, then clear out all retrieved data for object
            if not isinstance(spectrum[0], np.ndarray):
                spectrum = None

            if verbose: print('Read as FITS...')
        except (IOError, KeyError):
            # Check if the FITS file is just Numpy arrays
            try:
                spectrum, header = pf.getdata(downloaded_file, cache=True, header=True)
                if verbose: print('Read as FITS Numpy array...')
            except (IOError, KeyError):
                try:  # Try ascii
                    spectrum = ii.read(downloaded_file)
                    spectrum = np.array([np.asarray(spectrum.columns[n]) for n in range(len(spectrum.columns))])
                    if verbose: print('Read as ascii...')

                    txt, header = open(downloaded_file), []
                    for i in txt:
                        if any([i.startswith(char) for char in ['#', '|', '\\']]):
                            header.append(i.replace('\n', ''))
                    txt.close()
                except:
                    pass

    if spectrum == '':
        print('Could not retrieve spectrum at {}.'.format(File))
        return File
    else:
        spectrum = Spectrum(spectrum, header, File)
        return spectrum