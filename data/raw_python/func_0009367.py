def _converge(c):
    """
    Examine world state, returning data on what needs updating for release.

    :param c: Invoke ``Context`` object or subclass.

    :returns:
        Two dicts (technically, dict subclasses, which allow attribute access),
        ``actions`` and ``state`` (in that order.)

        ``actions`` maps release component names to variables (usually class
        constants) determining what action should be taken for that component:

        - ``changelog``: members of `.Changelog` such as ``NEEDS_RELEASE`` or
          ``OKAY``.
        - ``version``: members of `.VersionFile`.

        ``state`` contains the data used to calculate the actions, in case the
        caller wants to do further analysis:

        - ``branch``: the name of the checked-out Git branch.
        - ``changelog``: the parsed project changelog, a `dict` of releases.
        - ``release_type``: what type of release the branch appears to be (will
          be a member of `.Release` such as ``Release.BUGFIX``.)
        - ``latest_line_release``: the latest changelog release found for
          current release type/line.
        - ``latest_overall_release``: the absolute most recent release entry.
          Useful for determining next minor/feature release.
        - ``current_version``: the version string as found in the package's
          ``__version__``.
    """
    #
    # Data/state gathering
    #

    # Get data about current repo context: what branch are we on & what kind of
    # release does it appear to represent?
    branch, release_type = _release_line(c)
    # Short-circuit if type is undefined; we can't do useful work for that.
    if release_type is Release.UNDEFINED:
        raise UndefinedReleaseType(
            "You don't seem to be on a release-related branch; "
            "why are you trying to cut a release?"
        )
    # Parse our changelog so we can tell what's released and what's not.
    # TODO: below needs to go in something doc-y somewhere; having it in a
    # non-user-facing subroutine docstring isn't visible enough.
    """
    .. note::
        Requires that one sets the ``packaging.changelog_file`` configuration
        option; it should be a relative or absolute path to your
        ``changelog.rst`` (or whatever it's named in your project).
    """
    # TODO: allow skipping changelog if not using Releases since we have no
    # other good way of detecting whether a changelog needs/got an update.
    # TODO: chdir to sphinx.source, import conf.py, look at
    # releases_changelog_name - that way it will honor that setting and we can
    # ditch this explicit one instead. (and the docstring above)
    changelog = parse_changelog(
        c.packaging.changelog_file, load_extensions=True
    )
    # Get latest appropriate changelog release and any unreleased issues, for
    # current line
    line_release, issues = _release_and_issues(changelog, branch, release_type)
    # Also get latest overall release, sometimes that matters (usually only
    # when latest *appropriate* release doesn't exist yet)
    overall_release = _versions_from_changelog(changelog)[-1]
    # Obtain the project's main package & its version data
    current_version = load_version(c)
    # Grab all git tags
    tags = _get_tags(c)

    state = Lexicon(
        {
            "branch": branch,
            "release_type": release_type,
            "changelog": changelog,
            "latest_line_release": Version(line_release)
            if line_release
            else None,
            "latest_overall_release": overall_release,  # already a Version
            "unreleased_issues": issues,
            "current_version": Version(current_version),
            "tags": tags,
        }
    )
    # Version number determinations:
    # - latest actually-released version
    # - the next version after that for current branch
    # - which of the two is the actual version we're looking to converge on,
    # depends on current changelog state.
    latest_version, next_version = _latest_and_next_version(state)
    state.latest_version = latest_version
    state.next_version = next_version
    state.expected_version = latest_version
    if state.unreleased_issues:
        state.expected_version = next_version

    #
    # Logic determination / convergence
    #

    actions = Lexicon()

    # Changelog: needs new release entry if there are any unreleased issues for
    # current branch's line.
    # TODO: annotate with number of released issues [of each type?] - so not
    # just "up to date!" but "all set (will release 3 features & 5 bugs)"
    actions.changelog = Changelog.OKAY
    if release_type in (Release.BUGFIX, Release.FEATURE) and issues:
        actions.changelog = Changelog.NEEDS_RELEASE

    # Version file: simply whether version file equals the target version.
    # TODO: corner case of 'version file is >1 release in the future', but
    # that's still wrong, just would be a different 'bad' status output.
    actions.version = VersionFile.OKAY
    if state.current_version != state.expected_version:
        actions.version = VersionFile.NEEDS_BUMP

    # Git tag: similar to version file, except the check is existence of tag
    # instead of comparison to file contents. We even reuse the
    # 'expected_version' variable wholesale.
    actions.tag = Tag.OKAY
    if state.expected_version not in state.tags:
        actions.tag = Tag.NEEDS_CUTTING

    #
    # Return
    #

    return actions, state