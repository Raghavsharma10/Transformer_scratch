def load_plug_ins(app, root_dir):
    """Load plug-ins."""
    global extensions

    ui = app.ui

    # a list of directories in which sub-directories PlugIns will be searched.
    subdirectories = []

    # the default location is where the directory main packages are located.
    if root_dir:
        subdirectories.append(root_dir)

    # also search the default data location; create directory there if it doesn't exist to make it easier for user.
    # default data location will be application specific.
    data_location = ui.get_data_location()
    if data_location is not None:
        subdirectories.append(data_location)
        # create directories here if they don't exist
        plugins_dir = os.path.abspath(os.path.join(data_location, "PlugIns"))
        if not os.path.exists(plugins_dir):
            logging.info("Creating plug-ins directory %s", plugins_dir)
            os.makedirs(plugins_dir)

    # search the Nion/Swift subdirectory of the default document location too,
    # but don't create directories here - avoid polluting user visible directories.
    document_location = ui.get_document_location()
    if document_location is not None:
        subdirectories.append(os.path.join(document_location, "Nion", "Swift"))
        # do not create them in documents if they don't exist. this location is optional.

    # build a list of directories that will be loaded as plug-ins.
    PlugInDir = collections.namedtuple("PlugInDir", ["directory", "relative_path"])
    plugin_dirs = list()

    # track directories that have already been searched.
    seen_plugin_dirs = list()

    # for each subdirectory, look in PlugIns for sub-directories that represent the plug-ins.
    for subdirectory in subdirectories:
        plugins_dir = os.path.abspath(os.path.join(subdirectory, "PlugIns"))

        if os.path.exists(plugins_dir) and not plugins_dir in seen_plugin_dirs:
            logging.info("Loading plug-ins from %s", plugins_dir)

            # add the PlugIns directory to the system import path.
            sys.path.append(plugins_dir)

            # now build a list of sub-directories representing plug-ins within plugins_dir.
            sorted_relative_paths = sorted([d for d in os.listdir(plugins_dir) if os.path.isdir(os.path.join(plugins_dir, d))])
            plugin_dirs.extend([PlugInDir(plugins_dir, sorted_relative_path) for sorted_relative_path in sorted_relative_paths])

            # mark plugins_dir as 'seen' to avoid search it twice.
            seen_plugin_dirs.append(plugins_dir)
        else:
            logging.info("NOT Loading plug-ins from %s (missing)", plugins_dir)

    version_map = dict()
    module_exists_map = dict()

    plugin_adapters = list()

    import nionswift_plugin
    for module_info in pkgutil.iter_modules(nionswift_plugin.__path__):
        plugin_adapters.append(ModuleAdapter(nionswift_plugin.__name__, module_info))

    for directory, relative_path in plugin_dirs:
        plugin_adapters.append(PlugInAdapter(directory, relative_path))

    progress = True
    while progress:
        progress = False
        plugin_adapters_copy = copy.deepcopy(plugin_adapters)
        plugin_adapters = list()
        for plugin_adapter in plugin_adapters_copy:
            manifest_path = plugin_adapter.manifest_path
            manifest = plugin_adapter.manifest
            if manifest:
                manifest_valid = True
                if not "name" in manifest:
                    logging.info("Invalid manifest (missing 'name'): %s", manifest_path)
                    manifest_valid = False
                if not "identifier" in manifest:
                    logging.info("Invalid manifest (missing 'identifier'): %s", manifest_path)
                    manifest_valid = False
                if "identifier" in manifest and not re.match("[_\-a-zA-Z][_\-a-zA-Z0-9.]*$", manifest["identifier"]):
                    logging.info("Invalid manifest (invalid 'identifier': '%s'): %s", manifest["identifier"], manifest_path)
                    manifest_valid = False
                if not "version" in manifest:
                    logging.info("Invalid manifest (missing 'version'): %s", manifest_path)
                    manifest_valid = False
                if "requires" in manifest and not isinstance(manifest["requires"], list):
                    logging.info("Invalid manifest ('requires' not a list): %s", manifest_path)
                    manifest_valid = False
                if not manifest_valid:
                    continue
                for module in manifest.get("modules", list()):
                    if module in module_exists_map:
                        module_exists = module_exists_map.get(module)
                    else:
                        module_exists = importlib.util.find_spec(module) is not None
                        module_exists_map[module] = module_exists
                    if not module_exists:
                        logging.info("Plug-in '" + plugin_adapter.module_name + "' NOT loaded (" + plugin_adapter.module_path + ").")
                        logging.info("Cannot satisfy requirement (%s): %s", module, manifest_path)
                        manifest_valid = False
                        break
                for requirement in manifest.get("requires", list()):
                    # TODO: see https://packaging.pypa.io/en/latest/
                    requirement_components = requirement.split()
                    if len(requirement_components) != 3 or requirement_components[1] != "~=":
                        logging.info("Invalid manifest (requirement '%s' invalid): %s", requirement, manifest_path)
                        manifest_valid = False
                        break
                    identifier, operator, version_specifier = requirement_components[0], requirement_components[1], requirement_components[2]
                    if identifier in version_map:
                        if Utility.compare_versions("~" + version_specifier, version_map[identifier]) != 0:
                            logging.info("Plug-in '" + plugin_adapter.module_name + "' NOT loaded (" + plugin_adapter.module_path + ").")
                            logging.info("Cannot satisfy requirement (%s): %s", requirement, manifest_path)
                            manifest_valid = False
                            break
                    else:
                        # requirements not loaded yet; add back to plugin_adapters, but don't mark progress since nothing was loaded.
                        logging.info("Plug-in '" + plugin_adapter.module_name + "' delayed (%s) (" + plugin_adapter.module_path + ").", requirement)
                        plugin_adapters.append(plugin_adapter)
                        manifest_valid = False
                        break
                if not manifest_valid:
                    continue
                version_map[manifest["identifier"]] = manifest["version"]
            # read the manifests, if any
            # repeat loop of plug-ins until no plug-ins left in the list
            #   if all dependencies satisfied for a plug-in, load it
            #   otherwise defer until next round
            #   stop if no plug-ins loaded in the round
            #   count on the user to have correct dependencies
            module = plugin_adapter.load()
            if module:
                __modules.append(module)
            progress = True
    for plugin_adapter in plugin_adapters:
        logging.info("Plug-in '" + plugin_adapter.module_name + "' NOT loaded (requirements) (" + plugin_adapter.module_path + ").")

    notify_modules("run")