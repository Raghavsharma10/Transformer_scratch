def _run(
    categories, param_file, project_dir, plugin, target,
    status_update_interval
):
    """
    Generate code for this project and run it.
    Internal function we use for both run and flash commands.
    """
    project_dir = os.path.abspath(project_dir)

    # Make sure the project has been initialized
    pio_config = os.path.join(project_dir, "platformio.ini")
    if not os.path.isfile(pio_config):
        raise click.ClickException(
            "Not an OpenAg firmware project. To initialize a new project "
            "please use the `openag firmware init` command"
        )

    # @TODO in future we should pass in module config files as a thing
    # separate from the param file.
    name, ext = os.path.splitext(param_file.name)
    if ext == ".json":
        params = json.load(param_file)
    elif ext in (".yaml", ".yml"):
        params = yaml.load(param_file)
    else:
        raise ValueError("Param file must be YAML or JSON")

    firmware_type_params = params.get(FIRMWARE_MODULE_TYPE, [])
    firmware_params = params.get(FIRMWARE_MODULE, [])

    firmware_types = [
        FirmwareModuleType(record)
        for record in firmware_type_params
    ]
    firmware = [
        FirmwareModule(record)
        for record in firmware_params
    ]

    # Check for working modules in the lib folder
    # Do this second so project-local values overwrite values from the server
    lib_path = os.path.join(project_dir, "lib")
    for dir_name in os.listdir(lib_path):
        dir_path = os.path.join(lib_path, dir_name)
        if not os.path.isdir(dir_path):
            continue
        config_path = os.path.join(dir_path, "module.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                click.echo(
                    "Parsing firmware module type \"{}\" from lib "
                    "folder".format(dir_name)
                )
                doc = json.load(f)
                if not doc.get("_id"):
                    # Patch in id if id isn't present
                    doc["_id"] = parent_dirname(config_path)
                firmware_types.append(FirmwareModuleType(doc))

    if len(firmware) == 0:
        click.echo("Warning: no modules specified for the project")

    module_types = index_by_id(firmware_types)
    modules = index_by_id(firmware)
    # Synthesize the module and module type dicts
    modules = synthesize_firmware_module_info(modules, module_types)
    # Update the module inputs and outputs using the categories
    modules = prune_unspecified_categories(modules, categories)

    # Generate src.ino
    src_dir = os.path.join(project_dir, "src")
    src_file_path = os.path.join(src_dir, "src.ino")
    # Load the plugins
    plugin_fns = (load_plugin(plugin_name) for plugin_name in plugin)
    # Run modules through each plugin
    plugins = [plugin_fn(modules) for plugin_fn in plugin_fns]

    # Generate the code
    codegen = CodeGen(
        modules=modules, plugins=plugins,
        status_update_interval=status_update_interval
    )
    pio_ids = (dep["id"] for dep in codegen.all_pio_dependencies())
    for _id in pio_ids:
        subprocess.call(["platformio", "lib", "install", str(_id)])
    lib_dir = os.path.join(project_dir, "lib")
    for dep in codegen.all_git_dependencies():
        url = dep["url"]
        branch = dep.get("branch", "master")
        dep_folder_name = make_dir_name_from_url(url)
        dep_folder = os.path.join(lib_dir, dep_folder_name)
        if os.path.isdir(dep_folder):
            click.echo('Updating "{}"'.format(dep_folder_name))
            subprocess.call(
                ["git", "checkout", "--quiet", branch], cwd=dep_folder)
            subprocess.call(["git", "pull"], cwd=dep_folder)
        else:
            click.echo('Downloading "{}"'.format(dep_folder_name))
            subprocess.call(
                ["git", "clone", "-b", branch, url, dep_folder], cwd=lib_dir)
    with open(src_file_path, "w+") as f:
        codegen.write_to(f)

    # Compile the generated code
    command = ["platformio", "run"]
    if target:
        command.append("-t")
        command.append(target)
    env = os.environ.copy()
    build_flags = []
    for c in categories:
        build_flags.append("-DOPENAG_CATEGORY_{}".format(c.upper()))
    env["PLATFORMIO_BUILD_FLAGS"] = " ".join(build_flags)
    if subprocess.call(command, cwd=project_dir, env=env):
        raise click.ClickException("Compilation failed")