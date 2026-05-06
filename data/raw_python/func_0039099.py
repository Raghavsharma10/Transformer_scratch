def local(tool, slug, config_loader, offline=False):
    """
    Create/update local copy of github.com/org/repo/branch.
    Returns path to local copy
    """
    # Parse slug
    slug = Slug(slug, offline=offline)

    local_path = Path(LOCAL_PATH).expanduser() / slug.org / slug.repo

    git = Git(f"-C {shlex.quote(str(local_path))}")
    if not local_path.exists():
        _run(Git()(f"init {shlex.quote(str(local_path))}"))
        _run(git(f"remote add origin https://github.com/{slug.org}/{slug.repo}"))

    if not offline:
        # Get latest version of checks
        _run(git(f"fetch origin {slug.branch}"))

    # Ensure that local copy of the repo is identical to remote copy
    _run(git(f"checkout -f -B {slug.branch} origin/{slug.branch}"))
    _run(git(f"reset --hard HEAD"))

    problem_path = (local_path / slug.problem).absolute()

    if not problem_path.exists():
        raise InvalidSlugError(_("{} does not exist at {}/{}").format(slug.problem, slug.org, slug.repo))

    # Get config
    try:
        with open(problem_path / ".cs50.yaml") as f:
            try:
                config = config_loader.load(f.read())
            except InvalidConfigError:
                raise InvalidSlugError(
                    _("Invalid slug for {}. Did you mean something else?").format(tool))
    except FileNotFoundError:
        raise InvalidSlugError(_("Invalid slug. Did you mean something else?"))

    return problem_path