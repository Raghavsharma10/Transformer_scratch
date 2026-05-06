def pull_request(ctx, base_branch, open_pr, stop_timer):
    """Create a new pull request for this issue."""
    lancet = ctx.obj

    review_status = lancet.config.get("tracker", "review_status")
    remote_name = lancet.config.get("repository", "remote_name")

    if not base_branch:
        base_branch = lancet.config.get("repository", "base_branch")

    # Get the issue
    issue = get_issue(lancet)

    transition = get_transition(ctx, lancet, issue, review_status)

    # Get the working branch
    branch = get_branch(lancet, issue, create=False)

    with taskstatus("Checking pre-requisites") as ts:
        if not branch:
            ts.abort("No working branch found")

        if lancet.tracker.whoami() not in issue.assignees:
            ts.abort("Issue currently not assigned to you")

        # TODO: Check mergeability

    # TODO: Check remote status (PR does not already exist)

    # Push to remote
    with taskstatus('Pushing to "{}"', remote_name) as ts:
        remote = lancet.repo.lookup_remote(remote_name)
        if not remote:
            ts.abort('Remote "{}" not found', remote_name)

        from ..git import CredentialsCallbacks

        remote.push([branch.name], callbacks=CredentialsCallbacks())

        ts.ok('Pushed latest changes to "{}"', remote_name)

    # Create pull request
    with taskstatus("Creating pull request") as ts:
        template_path = lancet.config.get("repository", "pr_template")
        message = edit_template(template_path, issue=issue)

        if not message:
            ts.abort("You didn't provide a title for the pull request")

        title, body = message.split("\n", 1)
        title = title.strip()

        if not title:
            ts.abort("You didn't provide a title for the pull request")

        try:
            pr = lancet.scm_manager.create_pull_request(
                branch.branch_name, base_branch, title, body.strip("\n")
            )
        except PullRequestAlreadyExists as e:
            pr = e.pull_request
            ts.ok("Pull request does already exist at {}", pr.link)
        else:
            ts.ok("Pull request created at {}", pr.link)

    # Update issue
    set_issue_status(lancet, issue, review_status, transition)

    # TODO: Post to activity stream on JIRA?
    # TODO: Post to Slack?

    # Stop harvest timer
    if stop_timer:
        with taskstatus("Pausing harvest timer") as ts:
            lancet.timer.pause()
            ts.ok("Harvest timer paused")

    # Open the pull request page in the browser if requested
    if open_pr:
        click.launch(pr.link)