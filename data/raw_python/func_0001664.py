def workon(ctx, issue_id, new, base_branch):
    """
    Start work on a given issue.

    This command retrieves the issue from the issue tracker, creates and checks
    out a new aptly-named branch, puts the issue in the configured active,
    status, assigns it to you and starts a correctly linked Harvest timer.

    If a branch with the same name as the one to be created already exists, it
    is checked out instead. Variations in the branch name occuring after the
    issue ID are accounted for and the branch renamed to match the new issue
    summary.

    If the `default_project` directive is correctly configured, it is enough to
    give the issue ID (instead of the full project prefix + issue ID).
    """
    lancet = ctx.obj

    if not issue_id and not new:
        raise click.UsageError("Provide either an issue ID or the --new flag.")
    elif issue_id and new:
        raise click.UsageError(
            "Provide either an issue ID or the --new flag, but not both."
        )

    if new:
        # Create a new issue
        summary = click.prompt("Issue summary")
        issue = create_issue(
            lancet, summary=summary, add_to_active_sprint=True
        )
    else:
        issue = get_issue(lancet, issue_id)

    username = lancet.tracker.whoami()
    active_status = lancet.config.get("tracker", "active_status")
    if not base_branch:
        base_branch = lancet.config.get("repository", "base_branch")

    # Get the working branch
    branch = get_branch(lancet, issue, base_branch)

    # Make sure the issue is in a correct status
    transition = get_transition(ctx, lancet, issue, active_status)

    # Make sure the issue is assigned to us
    assign_issue(lancet, issue, username, active_status)

    # Activate environment
    set_issue_status(lancet, issue, active_status, transition)

    with taskstatus("Checking out working branch") as ts:
        lancet.repo.checkout(branch.name)
        ts.ok('Checked out working branch based on "{}"'.format(base_branch))

    with taskstatus("Starting harvest timer") as ts:
        lancet.timer.start(issue)
        ts.ok("Started harvest timer")