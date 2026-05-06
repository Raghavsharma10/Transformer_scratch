def resume(ctx):
    """
    Resume work on the currently active issue.

    The issue is retrieved from the currently active branch name.
    """
    lancet = ctx.obj

    username = lancet.tracker.whoami()
    active_status = lancet.config.get("tracker", "active_status")

    # Get the issue
    issue = get_issue(lancet)

    # Make sure the issue is in a correct status
    transition = get_transition(ctx, lancet, issue, active_status)

    # Make sure the issue is assigned to us
    assign_issue(lancet, issue, username, active_status)

    # Activate environment
    set_issue_status(lancet, issue, active_status, transition)

    with taskstatus("Resuming harvest timer") as ts:
        lancet.timer.start(issue)
        ts.ok("Resumed harvest timer")