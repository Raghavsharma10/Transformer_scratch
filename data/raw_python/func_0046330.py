def run(command, exit, silent, check):
    """
    Runs given command on all repos and checks status

        $ maintain repo run -- git checkout master
    """

    status = 0

    for (repo, path) in gather_repositories():
        if check and not check_repo(repo, path):
            status = 1

            if exit:
                break

            continue

        with chdir(path):
            result = subprocess.run(command, shell=True, capture_output=silent)
            if result.returncode != 0:
                status = result.returncode

                print('Command failed: {}'.format(repo))

                if exit:
                    break

    sys.exit(status)