def run(files, temp_folder, arg=None):
    "Check we're not committing to a blocked branch"
    parser = get_parser()
    argos = parser.parse_args(arg.split())

    current_branch = bash('git symbolic-ref HEAD').value()
    current_branch = current_branch.replace('refs/heads/', '').strip()
    if current_branch in argos.branches:
        return ("Branch '{0}' is blocked from being "
                "committed to.".format(current_branch))