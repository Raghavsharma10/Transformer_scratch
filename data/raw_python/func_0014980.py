def default_hook(config):
    """Default setup hook."""
    if (any(arg.startswith('bdist') for arg in sys.argv) and
            os.path.isdir(PY2K_DIR) != IS_PY2K and os.path.isdir(LIB_DIR)):
        shutil.rmtree(LIB_DIR)

    if IS_PY2K and any(arg.startswith('install') or
                       arg.startswith('build') or
                       arg.startswith('bdist') for arg in sys.argv):
        generate_py2k(config)
        packages_root = get_cfg_value(config, 'files', 'packages_root')
        packages_root = os.path.join(PY2K_DIR, packages_root)
        set_cfg_value(config, 'files', 'packages_root', packages_root)