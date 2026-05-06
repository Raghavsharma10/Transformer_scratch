def alter_poms(pom_dir, additional_params, repo_url=None, mvn_repo_local=None):
    """
    Runs mvn clean command with provided additional parameters to perform pom updates by pom-manipulation-ext.
    """
    work_dir = os.getcwd()
    os.chdir(pom_dir)

    try:
        if repo_url:
            settings_filename = create_mirror_settings(repo_url)
        else:
            settings_filename = None

        args = ["mvn", "clean"]
        if mvn_repo_local:
            args.extend(["-s", settings_filename])
        if mvn_repo_local:
            args.append("-Dmaven.repo.local=%s" % mvn_repo_local)
        param_list = additional_params.split(" ")
        args.extend(param_list)

        logging.debug("Running command: %s", " ".join(args))
        command = Popen(args, stdout=PIPE, stderr=STDOUT)
        stdout = command.communicate()[0]
        if command.returncode:
            logging.error("POM manipulation failed. Output:\n%s" % stdout)
        else:
            logging.debug("POM manipulation succeeded. Output:\n%s" % stdout)
    finally:
        os.chdir(work_dir)