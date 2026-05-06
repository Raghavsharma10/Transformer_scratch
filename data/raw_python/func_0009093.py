def run_build(build_dir, params, verbose=True):
    '''run_build takes a build directory and params dictionary, and does the following:
      - downloads repo to a temporary directory
      - changes branch or commit, if needed
      - creates and bootstraps singularity image from Singularity file
      - returns a dictionary with: 
          image (path), metadata (dict)

    The following must be included in params: 
       spec_file, repo_url, branch, commit

    '''

    # Download the repository

    download_repo(repo_url=params['repo_url'],
                  destination=build_dir)

    os.chdir(build_dir)

    if params['branch'] != None:
        bot.info('Checking out branch %s' %params['branch'])
        os.system('git checkout %s' %(params['branch']))
    else:
        params['branch'] = "master"


    # Set the debug level

    Client.debug = params['debug']

    # Commit

    if params['commit'] not in [None,'']:
        bot.info('Checking out commit %s' %params['commit'])
        os.system('git checkout %s .' %(params['commit']))

    # From here on out commit is used as a unique id, if we don't have one, we use current
    else:
        params['commit'] = os.popen('git log -n 1 --pretty=format:"%H"').read()
        bot.warning("commit not specified, setting to current %s" %params['commit'])

    # Dump some params for the builder, in case it fails after this
    passing_params = "/tmp/params.pkl"
    pickle.dump(params, open(passing_params,'wb'))

    # Now look for spec file
    if os.path.exists(params['spec_file']):
        bot.info("Found spec file %s in repository" %params['spec_file'])

        # If the user has a symbolic link
        if os.path.islink(params['spec_file']):
            bot.info("%s is a symbolic link." %params['spec_file'])
            params['spec_file'] = os.path.realpath(params['spec_file'])

        # START TIMING
        start_time = datetime.now()

        # Secure Build
        image = Client.build(recipe=params['spec_file'],
                             build_folder=build_dir,
                             isolated=True)

        # Save has for metadata (also is image name)
        version = get_image_file_hash(image)
        params['version'] = version
        pickle.dump(params, open(passing_params,'wb'))

        # Rename image to be hash
        finished_image = "%s/%s.simg" %(os.path.dirname(image), version)
        image = shutil.move(image, finished_image)

        final_time = (datetime.now() - start_time).seconds
        bot.info("Final time of build %s seconds." %final_time)  

        # Did the container build successfully?
        test_result = test_container(image)
        if test_result['return_code'] != 0:
            bot.error("Image failed to build, cancelling.")
            sys.exit(1)

        # Get singularity version
        singularity_version = Client.version()
        Client.debug = False
        inspect = Client.inspect(image) # this is a string
        Client.debug = params['debug']

        # Get information on apps
        Client.debug = False
        app_names = Client.apps(image)
        Client.debug = params['debug']
        apps = extract_apps(image, app_names)
        
        metrics = {'build_time_seconds': final_time,
                   'singularity_version': singularity_version,
                   'singularity_python_version': singularity_python_version, 
                   'inspect': inspect,
                   'version': version,
                   'apps': apps}
  
        output = {'image':image,
                  'metadata':metrics,
                  'params':params }

        return output

    else:

        # Tell the user what is actually there
        present_files = glob("*")
        bot.error("Build file %s not found in repository" %params['spec_file'])
        bot.info("Found files are %s" %"\n".join(present_files))
        # Params have been exported, will be found by log
        sys.exit(1)