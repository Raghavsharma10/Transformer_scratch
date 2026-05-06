def download_files(manifest, download_dir, auth_token=None,
                       chunk_size=1048576, avoid_redownload=True):
    """Individually download files from GDC.
    
    Params
    ------
    manifest : `pandas.DataFrame`
        GDC manifest that contains a list of files. The data frame should
        have five columns: id, filename, md5, size, and state.
    download_dir : str
        The path of the download directory.
    auth_token : str, optional
        Authentication token for downloading protected data.
        If None, do not send authentication header. [None]
    chunk_size : int, optional
        The chunk size (in bytes) to use for downloading data. [1048576]
        
    Returns
    -------
    None
    """
    assert isinstance(manifest, pd.DataFrame)
    assert isinstance(download_dir, str)
    assert isinstance(chunk_size, int)
    if auth_token is not None:
        assert isinstance(auth_token, str)
                       
    def get_file_md5hash(path):
        """Calculate the MD5 hash for a file."""
        with open(path, 'rb') as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        return h

    headers = {}
    if auth_token is not None:
        headers['X-Auth-Token'] = auth_token
        
    #payload = {'ids': file_ids}    
    #logger.info('Downloading data to "%s"...', download_dir)
    num_files = manifest.shape[0]
    logger.info('Downloading %d files to "%s"...', num_files, download_dir)
    for i, row in manifest.iterrows():
        #(uuid, (file_name, file_hash)) 
        success = False        
        download_file = os.path.join(download_dir, row['filename'])
        if ((i+1) % 100) == 0:
            logger.info('Downloading file %d / %d...', i+1, num_files)
        
        if avoid_redownload and os.path.isfile(download_file) and \
                get_file_md5hash(download_file) == row['md5']:
            logger.info('File %s already downloaded...skipping.',
                        download_file)
            success = True
        
        while not success:
            with closing(
                requests.get('https://gdc-api.nci.nih.gov/data/%s'
                             % row['id'], headers=headers, stream=True)) \
                as r:
                # get suggested file name from "Content-Disposition" header
                # suggested_file_name = re.findall(
                #       "filename=(.+)", r.headers['Content-Disposition'])[0]

                r.raise_for_status()

                with open(download_file, 'wb') as ofh:
                    for chunk in r.iter_content(chunk_size=chunk_size): 
                        if chunk: # filter out keep-alive new chunks
                            ofh.write(chunk)
            with open(download_file, 'rb') as fh:
                h = hashlib.md5(fh.read()).hexdigest()
                if h == row['md5']:
                    success = True
            if not success:
                logger.warning('Hash value mismatch (should be: %s; is: %s). '
                               'Attempting to re-download file...',
                               row['md5'], h)