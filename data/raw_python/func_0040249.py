def build(args):
    """Build a target and its dependencies."""

    if len(args) != 1:
        log.error('One target required.')
        app.quit(1)

    target = address.new(args[0])
    log.info('Resolved target to: %s', target)

    try:
        bb = Butcher()
        bb.clean()
        bb.load_graph(target)
        bb.build(target)
    except (gitrepo.GitError,
            error.BrokenGraph,
            error.NoSuchTargetError) as err:
        log.fatal(err)
        app.quit(1)
    except error.OverallBuildFailure as err:
        log.fatal(err)
        log.fatal('Error list:')
        [log.fatal('  [%s]: %s', e.node, e) for e in bb.failure_log]
        app.quit(1)