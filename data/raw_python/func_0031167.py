def get_annotation_urls_and_checksums(species, release=None, ftp=None):
    """Get FTP URLs and checksums for Ensembl genome annotations.
    
    Parameters
    ----------
    species : str or list of str
        The species or list of species for which to get genome annotations
        (e.g., "Homo_sapiens").
    release : int, optional
        The release number to look up. If `None`, use latest release. [None]
    ftp : ftplib.FTP, optional
        The FTP connection to use. If `None`, the function will open and close
        its own connection using user "anonymous".
    """
    ### type checks
    assert isinstance(species, (str, _oldstr)) or isinstance(species, Iterable)
    if release is not None:
        assert isinstance(release, int)
    if ftp is not None:
        assert isinstance(ftp, ftplib.FTP)

    ### open FTP connection if necessary
    close_connection = False
    ftp_server = 'ftp.ensembl.org'
    ftp_user = 'anonymous'
    if ftp is None:
        ftp = ftplib.FTP(ftp_server)
        ftp.login(ftp_user)
        close_connection = True    

    ### determine release if necessary
    if release is None:
        # use latest release
        release = util.get_latest_release(ftp=ftp)

    species_data = OrderedDict()
    if isinstance(species, (str, _oldstr)):
        species_list = [species]
    else:
        species_list = species
    for spec in species_list:

        # get the GTF file URL
        # => since the naming scheme isn't consistent across species,
        #    we're using a flexible scheme here to find the right file
        species_dir = '/pub/release-%d/gtf/%s' % (release, spec.lower())
        data = []
        ftp.dir(species_dir, data.append)
        gtf_file = []
        for d in data:
            i = d.rindex(' ')
            fn = d[(i + 1):]
            if fn.endswith('.%d.gtf.gz' % release):
                gtf_file.append(fn)
        assert len(gtf_file) == 1
        gtf_file = gtf_file[0]
        _LOGGER.debug('GTF file: %s', gtf_file)

        ### get the checksum for the GTF file
        checksum_url = '/'.join([species_dir, 'CHECKSUMS'])
        file_checksums = util.get_file_checksums(checksum_url, ftp=ftp)
        gtf_checksum = file_checksums[gtf_file]
        _LOGGER.debug('GTF file checksum: %d', gtf_checksum)

        gtf_url = 'ftp://%s%s/%s' %(ftp_server, species_dir, gtf_file)

        species_data[spec] = (gtf_url, gtf_checksum)

    # close FTP connection, if we opened it
    if close_connection:
        ftp.close()

    return species_data