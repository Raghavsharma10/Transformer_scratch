def main(commit_only=True):
    """
    Run the configured code checks.

    Return system exit code.
        1 - reject commit
        0 - accept commit
    """
    global TEMP_FOLDER
    exit_code = 0
    hook_checks = HookConfig(get_config_file())
    with files_to_check(commit_only) as files:
        for name, mod in checks():
            default = getattr(mod, 'DEFAULT', 'off')
            if hook_checks.is_enabled(name, default=default):
                if hasattr(mod, 'REQUIRED_FILES'):
                    for filename in mod.REQUIRED_FILES:
                        if os.path.isfile(filename):
                            try:
                                shutil.copy(filename, TEMP_FOLDER)
                            except shutil.Error:
                                # Copied over by a previous check
                                continue
                args = hook_checks.arguments(name)

                tmp_files = [os.path.join(TEMP_FOLDER, f) for f in files]

                if args:
                    errors = mod.run(tmp_files, TEMP_FOLDER, args)
                else:
                    errors = mod.run(tmp_files, TEMP_FOLDER)
                if errors:
                    title_print("Checking {0}".format(name))
                    print((errors.replace(TEMP_FOLDER + "/", '')))
                    print("")
                    exit_code = 1

    if exit_code == 1:
        title_print("Rejecting commit")
    return exit_code