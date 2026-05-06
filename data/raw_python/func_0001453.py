def bump(args: argparse.Namespace) -> None:
    """
    :args: An argparse.Namespace object.

    This function is bound to the 'bump' sub-command. It increments the version
    integer of the user's choice ('major', 'minor', or 'patch').
    """
    try:
        last_tag = last_git_release_tag(git_tags())
    except NoGitTagsException:
        print(SemVer(0, 1, 0))
        exit(0)

    last_ver = git_tag_to_semver(last_tag)

    if args.type == 'patch':
        print(last_ver.bump_patch())
    elif args.type == 'minor':
        print(last_ver.bump_minor())
    elif args.type == 'major':
        print(last_ver.bump_major())