def pause(ctx):
    """
    Pause work on the current issue.

    This command puts the issue in the configured paused status and stops the
    current Harvest timer.
    """
    lancet = ctx.obj
    paused_status = lancet.config.get("tracker", "paused_status")

    # Get the issue
    issue = get_issue(lancet)

    # Make sure the issue is in a correct status
    transition = get_transition(ctx, lancet, issue, paused_status)

    # Activate environment
    set_issue_status(lancet, issue, paused_status, transition)

    with taskstatus("Pausing harvest timer") as ts:
        lancet.timer.pause()
        ts.ok("Harvest timer paused")