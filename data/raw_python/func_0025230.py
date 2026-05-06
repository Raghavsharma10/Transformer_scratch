def bootstrap_main(args):
    """
    Main function explicitly called from the C++ code.
    Return the main application object.
    """
    version_info = sys.version_info
    if version_info.major != 3 or version_info.minor < 6:
        return None, "python36"
    main_fn = load_module_as_package("nionui_app.nionswift")
    if main_fn:
        return main_fn(["nionui_app.nionswift"] + args, {"pyqt": None}), None
    return None, "main"