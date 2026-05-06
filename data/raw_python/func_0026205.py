def convertwaiveredfits(waiveredObject,
                        outputFileName=None,
                        forceFileOutput=False,
                        convertTo='multiExtension',
                        verbose=False):
    """
        Convert the input waivered FITS object to various formats.  The
        default conversion format is multi-extension FITS.  Generate an output
        file in the desired format if requested.

        Parameters:

          waiveredObject  input object representing a waivered FITS file;
                          either a astropy.io.fits.HDUList object, a file object, or a
                          file specification

          outputFileName  file specification for the output file
                          Default: None - do not generate an output file

          forceFileOutput force the generation of an output file when the
                          outputFileName parameter is None; the output file
                          specification will be the same as the input file
                          specification with the last character of the base
                          name replaced with the character `h` in
                          multi-extension FITS format.

                          Default: False

          convertTo       target conversion type
                          Default: 'multiExtension'

          verbose         provide verbose output
                          Default: False

        Returns:

          hdul            an HDUList object in the requested format.

        Exceptions:

           ValueError       Conversion type is unknown
    """

    if convertTo == 'multiExtension':
        func = toMultiExtensionFits
    else:
        raise ValueError('Conversion type ' + convertTo + ' unknown')

    return func(*(waiveredObject,outputFileName,forceFileOutput,verbose))