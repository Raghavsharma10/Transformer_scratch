def convert_to_env_data(mgr, env_paths, validator_func, activate_func,
                        name_template, display_name_template, name_prefix):
    """Converts a list of paths to environments to env_data.

    env_data is a structure {name -> (ressourcedir, kernel spec)}
    """
    env_data = {}
    for venv_dir in env_paths:
        venv_name = os.path.split(os.path.abspath(venv_dir))[1]
        kernel_name = name_template.format(name_prefix + venv_name)
        kernel_name = kernel_name.lower()
        if kernel_name in env_data:
            mgr.log.debug(
                "Found duplicate env kernel: %s, which would again point to %s. Using the first!",
                kernel_name, venv_dir)
            continue
        argv, language, resource_dir = validator_func(venv_dir)
        if not argv:
            # probably does not contain the kernel type (e.g. not R or python or does not contain
            # the kernel code itself)
            continue
        display_name = display_name_template.format(kernel_name)
        kspec_dict = {"argv": argv, "language": language,
                      "display_name": display_name,
                      "resource_dir": resource_dir
                      }

        # the default vars are needed to save the vars in the function context
        def loader(env_dir=venv_dir, activate_func=activate_func, mgr=mgr):
            mgr.log.debug("Loading env data for %s" % env_dir)
            res = activate_func(mgr, env_dir)
            # mgr.log.info("PATH: %s" % res['PATH'])
            return res

        kspec = EnvironmentLoadingKernelSpec(loader, **kspec_dict)
        env_data.update({kernel_name: (resource_dir, kspec)})
    return env_data