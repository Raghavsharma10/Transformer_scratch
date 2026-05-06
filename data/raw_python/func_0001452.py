def infer(args: argparse.Namespace) -> None:
    """
    :args: An argparse.Namespace object.

    This is the function called when the 'infer' sub-command is passed as an
    argument to the CLI.
    """
    try:
        last_tag = last_git_release_tag(git_tags())
    except NoGitTagsException:
        print(SemVer(0, 1, 0))
        exit(0)

    commit_log = git_commits_since_last_tag(last_tag)
    action = parse_commit_log(commit_log)

    last_ver = git_tag_to_semver(last_tag)

    if action == 'min':
        new_ver = last_ver.bump_minor()
    elif action == 'maj':
        new_ver = last_ver.bump_major()
    else:
        new_ver = last_ver.bump_patch()

    print(new_ver)