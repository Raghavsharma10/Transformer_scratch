def finish_build(verbose=True):
    '''finish_build will finish the build by way of sending the log to the same bucket.
    the params are loaded from the previous function that built the image, expected in
    $HOME/params.pkl
    :: note: this function is currently configured to work with Google Compute
    Engine metadata api, and should (will) be customized if needed to work elsewhere 
    '''
    # If we are building the image, this will not be set
    go = get_build_metadata(key='dobuild')
    if go == None:
        sys.exit(0)

    # Load metadata
    passing_params = "/tmp/params.pkl"
    params = pickle.load(open(passing_params,'rb'))

    # Start the storage service, retrieve the bucket
    storage_service = get_google_service()
    bucket = get_bucket(storage_service,params['bucket_name'])

    # If version isn't in params, build failed
    version = 'error-%s' % str(uuid.uuid4())
    if 'version' in params:
        version = params['version']
    trailing_path = "%s/%s" %(params['commit'], version)
    image_path = get_image_path(params['repo_url'], trailing_path) 

    # Upload the log file
    params['log_file'] = upload_file(storage_service,
                                     bucket=bucket,
                                     bucket_path=image_path,
                                     file_name=params['logfile'])
                
    # Close up shop
    send_build_close(params=params,
                     response_url=params['logging_url'])