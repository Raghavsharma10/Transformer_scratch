def get_project_root(cls):
        '''
        Return path of the project directory.  Use this method for getting paths relative to the project.
        However, for data, it's recommended you use WTF_DATA_MANAGER and for assets it's recommended 
        to use WTF_ASSET_MANAGER, which are already singleton instances that manger the /data, and /assets 
        folder for you.

        Returns:
            str - path of project root directory.
        '''
        if(cls.__root_folder__ != None):
            return cls.__root_folder__

        # Check for enviornment variable override.
        try:
            cls.__root_folder__ = os.environ[cls.WTF_HOME_CONFIG_ENV_VAR]
        except KeyError:
            # Means WTF_HOME override isn't specified.
            pass

        # Search starting from the current working directory and traverse up parent directories for the
        # hidden file denoting the project root folder.
        path = os.getcwd()
        seperator_matches = re.finditer("/|\\\\", path)

        paths_to_search = [path]
        for match in seperator_matches:
            p = path[:match.start()]
            paths_to_search.insert(0, p)

        for path in paths_to_search:
            target_path = os.path.join(path, cls.__WTF_ROOT_FOLDER_FILE)
            if os.path.isfile(target_path):
                cls.__root_folder__ = path
                return cls.__root_folder__

        raise RuntimeError(u("Missing root project folder locator file '.wtf_root_folder'.")
                           + u("Check to make sure this file is located in your project directory."))