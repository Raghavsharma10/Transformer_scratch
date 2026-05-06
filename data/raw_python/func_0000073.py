def milestones(ctx, list, close):
    """View/edit/close milestones on github
    """
    repos = get_repos(ctx.parent.agile.get('labels'))
    if list:
        _list_milestones(repos)
    elif close:
        click.echo('Closing milestones "%s"' % close)
        _close_milestone(repos, close)
    else:
        click.echo(ctx.get_help())