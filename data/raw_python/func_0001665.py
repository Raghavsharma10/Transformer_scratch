def time(lancet, issue):
    """
    Start an Harvest timer for the given issue.

    This command takes care of linking the timer with the issue tracker page
    for the given issue. If the issue is not passed to command it's taken
    from currently active branch.
    """
    issue = get_issue(lancet, issue)

    with taskstatus("Starting harvest timer") as ts:
        lancet.timer.start(issue)
        ts.ok("Started harvest timer")