def issue_add(lancet, assign, add_to_sprint, summary):
    """
    Create a new issue on the issue tracker.
    """
    summary = " ".join(summary)
    issue = create_issue(
        lancet,
        summary,
        # project_id=project_id,
        add_to_active_sprint=add_to_sprint,
    )
    if assign:
        if assign == "me":
            username = lancet.tracker.whoami()
        else:
            username = assign
        assign_issue(lancet, issue, username)

    click.echo("Created issue")