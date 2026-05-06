def get_script(script_name):
    '''get_script will return a build script_name, if it is included 
    in singularity/build/scripts, otherwise will alert the user and return None
    :param script_name: the name of the script to look for
    '''
    install_dir = get_installdir()
    script_path = "%s/build/scripts/%s" %(install_dir,script_name)
    if os.path.exists(script_path):
        return script_path
    else:
        bot.error("Script %s is not included in singularity-python!" %script_path)
        return None