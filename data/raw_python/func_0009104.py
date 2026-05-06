def run_build(logfile='/tmp/.shub-log'):

    '''run_build will generate the Singularity build from a spec_file from a repo_url.

    If no arguments are required, the metadata api is queried for the values.

    :param build_dir: directory to do the build in. If not specified, will use temporary.   
    :param spec_file: the spec_file name to use, assumed to be in git repo
    :param repo_url: the url to download the repo from
    :param repo_id: the repo_id to uniquely identify the repo (in case name changes)
    :param commit: the commit to checkout. If none provided, will use most recent.
    :param bucket_name: the name of the bucket to send files to
    :param verbose: print out extra details as we go (default True)    
    :param token: a token to send back to the server to authenticate the collection
    :param secret: a secret to match to the correct container
    :param response_url: the build url to send the response back to. Should also come
    from metadata. If not specified, no response is sent
    :param branch: the branch to checkout for the build.

    :: note: this function is currently configured to work with Google Compute
    Engine metadata api, and should (will) be customized if needed to work elsewhere 

    '''

    # If we are building the image, this will not be set
    go = get_build_metadata(key='dobuild')
    if go == None:
        sys.exit(0)

    # If the user wants debug, this will be set
    debug = True
    enable_debug = get_build_metadata(key='debug')
    if enable_debug == None:
        debug = False
    bot.info('DEBUG %s' %debug)

    # Uaw /tmp for build directory
    build_dir = tempfile.mkdtemp()

    # Get variables from the instance metadata API
    metadata = [{'key': 'repo_url', 'value': None },
                {'key': 'repo_id', 'value': None },
                {'key': 'response_url', 'value': None },
                {'key': 'bucket_name', 'value': None },
                {'key': 'tag', 'value': None },
                {'key': 'container_id', 'value': None },
                {'key': 'commit', 'value': None },
                {'key': 'token', 'value': None},
                {'key': 'branch', 'value': None },
                {'key': 'spec_file', 'value': None},
                {'key': 'logging_url', 'value': None },
                {'key': 'logfile', 'value': logfile }]

    # Obtain values from build
    bot.log("BUILD PARAMETERS:")
    params = get_build_params(metadata)
    params['debug'] = debug
    
    # Default spec file is Singularity
    if params['spec_file'] == None:
        params['spec_file'] = "Singularity"
        
    if params['bucket_name'] == None:
        params['bucket_name'] = "singularityhub"

    if params['tag'] == None:
        params['tag'] = "latest"

    output = run_build_main(build_dir=build_dir,
                            params=params)

    # Output includes:
    finished_image = output['image']
    metadata = output['metadata']
    params = output['params']  

    # Upload image package files to Google Storage
    if os.path.exists(finished_image):
        bot.info("%s successfully built" %finished_image)
        dest_dir = tempfile.mkdtemp(prefix='build')

        # The path to the images on google drive will be the github url/commit folder
        trailing_path = "%s/%s" %(params['commit'], params['version'])
        image_path = get_image_path(params['repo_url'], trailing_path) 
                                                         # commits are no longer unique
                                                         # storage is by commit

        build_files = [finished_image]
        bot.info("Sending image to storage:") 
        bot.info('\n'.join(build_files))

        # Start the storage service, retrieve the bucket
        storage_service = get_google_service() # default is "storage" "v1"
        bucket = get_bucket(storage_service,params["bucket_name"])

        # For each file, upload to storage
        files = []
        for build_file in build_files:
            bot.info("Uploading %s to storage..." %build_file)
            storage_file = upload_file(storage_service,
                                       bucket=bucket,
                                       bucket_path=image_path,
                                       file_name=build_file)  
            files.append(storage_file)
                
        # Finally, package everything to send back to shub
        response = {"files": json.dumps(files),
                    "repo_url": params['repo_url'],
                    "commit": params['commit'],
                    "repo_id": params['repo_id'],
                    "branch": params['branch'],
                    "tag": params['tag'],
                    "container_id": params['container_id'],
                    "spec_file":params['spec_file'],
                    "token": params['token'],
                    "metadata": json.dumps(metadata)}

        # Did the user specify a specific log file?
        custom_logfile = get_build_metadata('logfile')
        if custom_logfile is not None:
            logfile = custom_logfile    
        response['logfile'] = logfile

        # Send final build data to instance
        send_build_data(build_dir=build_dir,
                        response_url=params['response_url'],
                        secret=params['token'],
                        data=response)

        # Dump final params, for logger to retrieve
        passing_params = "/tmp/params.pkl"
        pickle.dump(params,open(passing_params,'wb'))