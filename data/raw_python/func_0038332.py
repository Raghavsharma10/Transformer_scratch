def convert_image(File, verbose=False):
    """
    Converts a IMAGE data type stored in the database into a data cube

    Parameters
    ----------
    File: str
        The URL or filepath of the file to be converted into arrays.
    verbose: bool
        Whether or not to display some diagnostic information (Default: False)

    Returns
    -------
    sequence
        The converted image

    """
    image, header = '', ''
    if isinstance(File, type(b'')):  # Decode if needed (ie, for Python 3)
        File = File.decode('utf-8')
        
    if isinstance(File, (str, type(u''))):
        
        # Convert variable path to absolute path
        if File.startswith('$'):
            abspath = os.popen('echo {}'.format(File.split('/')[0])).read()[:-1]
            if abspath:
                File = File.replace(File.split('/')[0], abspath)
                
        if File.startswith('http'):
            if verbose:
                print('Downloading {}'.format(File))
            
            # Download only once
            downloaded_file = download_file(File, cache=True)
        else:
            downloaded_file = File
            
        try: 
            # Get the data
            image, header = pf.getdata(downloaded_file, cache=True, header=True)
            
            # If no data, then clear out all retrieved info for object
            if not isinstance(image, np.ndarray):
                image = None
                
            if verbose:
                print('Read as FITS...')
            
        except (IOError, KeyError):
            # Check if the FITS file is just Numpy arrays
            try:
                image, header = pf.getdata(downloaded_file, cache=True, header=True)
                if verbose:
                    print('Read as FITS Numpy array...')
                    
            except (IOError, KeyError):
                try:  # Try ascii
                    image = ii.read(downloaded_file)
                    image = np.array([np.asarray(image.columns[n]) for n in range(len(image.columns))])
                    if verbose:
                        print('Read as ascii...')
                        
                    txt, header = open(downloaded_file), []
                    for i in txt:
                        if any([i.startswith(char) for char in ['#', '|', '\\']]):
                            header.append(i.replace('\n', ''))
                    txt.close()
                    
                except:
                    pass
                    
    if image == '':
        print('Could not retrieve image at {}.'.format(File))
        return File
        
    else:
        image = Image(image, header, File)
        return image