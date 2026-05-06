def run_module(
    ctx, arguments, project_dir, board, **kwargs
):
    """
    Run a single instance of this module. [ARGUMENTS] specifies a list of
    implementation-specific arguments to the module (for example, configuring
    Arduino pin numbers for the module).

    Example:

    \b
    openag firmware run_module -t upload 4

    This command fetches module definitions from CouchDB. CouchDB must be
    running on port 5984 and the firmware_module_type database populated with
    appropriate type records for this command to work. Loading the default
    fixture from openag_brain will populate a default set of
    firmware_module_type records.
    """
    # Read the module config
    here = os.path.abspath(project_dir)
    module_json_path = os.path.join(here, "module.json")
    try:
        with open(module_json_path) as f:
            doc = json.load(f)
            if not doc.get("_id"):
                # Patch in id if not present
                doc["_id"] = parent_dirname(module_json_path)
            module_type = FirmwareModuleType(doc)
    except IOError:
        raise click.ClickException("No module.json file found")

    # Create the build directory
    build_path = os.path.join(here, "_build")
    if not os.path.isdir(build_path):
        os.mkdir(build_path)
    kwargs["project_dir"] = build_path

    # Initialize an openag project in the build directory
    ctx.invoke(init, board=board, **kwargs)

    # Link the source files into the lib directory
    lib_path = os.path.join(build_path, "lib")
    module_path = os.path.join(lib_path, "module")
    if not os.path.isdir(module_path):
        os.mkdir(module_path)
    for file_name in os.listdir(here):
        file_path = os.path.join(here, file_name)
        if not os.path.isfile(file_path) or file_name.startswith("."):
            continue
        source = "../../../{}".format(file_name)
        link_name = os.path.join(module_path, file_name)
        if os.path.isfile(link_name):
            os.remove(link_name)
        os.symlink(source, link_name)

    # Parse the arguments based on the module type
    real_args = []
    for i in range(len(arguments)):
        if i >= len(module_type["arguments"]):
            raise click.ClickException(
                "Too many module arguments specified. (Got {}, expected "
                "{})".format(len(arguments), len(module_type["arguments"]))
            )
        val = arguments[i]
        arg_info = module_type["arguments"][i]
        if arg_info["type"] == "int":
            val = int(val)
        elif arg_info["type"] == "float":
            val = float(val)
        elif arg_info["type"] == "bool":
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            else:
                raise click.BadParameter(
                    "Argument number {} should be a boolean value "
                    '("true" or "false")'.format(i)
                )
        real_args.append(val)

    # Write the modules.json file
    modules = {
        FIRMWARE_MODULE: [
            FirmwareModule({
                "_id": "module_1",
                "type": "module",
                "arguments": list(real_args)
            })
        ]
    }
    modules_file = os.path.join(build_path, "modules.json")
    with open(modules_file, "w") as f:
        json.dump(modules, f)
    with open(modules_file, "r") as f:
        kwargs["param_file"] = f
        # Run the project
        ctx.invoke(run, **kwargs)