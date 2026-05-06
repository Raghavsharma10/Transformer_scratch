def lifecycle(awsclient, env, tool, command, arguments):
    """Tool lifecycle which provides hooks into the different stages of the
    command execution. See signals for hook details.
    """
    log.debug('### init')
    load_plugins()
    context = get_context(awsclient, env, tool, command, arguments)
    # every tool needs a awsclient so we provide this via the context
    context['_awsclient'] = awsclient
    log.debug('### context:')
    log.debug(context)
    if 'error' in context:
        # no need to send an 'error' signal here
        return 1

    ## initialized
    gcdt_signals.initialized.send(context)
    log.debug('### initialized')
    if 'error' in context:
        log.error(context['error'])
        return 1
    check_gcdt_update()

    # config is "assembled" by config_reader NOT here!
    config = {}

    gcdt_signals.config_read_init.send((context, config))
    log.debug('### config_read_init')
    gcdt_signals.config_read_finalized.send((context, config))
    log.debug('### config_read_finalized')
    # TODO we might want to be able to override config via env variables?
    # here would be the right place to do this
    if 'hookfile' in config:
        # load hooks from hookfile
        _load_hooks(config['hookfile'])
    if 'kumo' in config:
        # deprecated: this needs to be removed once all old-style "cloudformation" entries are gone
        fix_old_kumo_config(config)

    # check_credentials
    gcdt_signals.check_credentials_init.send((context, config))
    log.debug('### check_credentials_init')
    gcdt_signals.check_credentials_finalized.send((context, config))
    log.debug('### check_credentials_finalized')
    if 'error' in context:
        log.error(context['error'])
        gcdt_signals.error.send((context, config))
        return 1

    ## lookup
    gcdt_signals.lookup_init.send((context, config))
    log.debug('### lookup_init')
    gcdt_signals.lookup_finalized.send((context, config))
    log.debug('### lookup_finalized')
    log.debug('### config after lookup:')
    log.debug(config)

    ## config validation
    gcdt_signals.config_validation_init.send((context, config))
    log.debug('### config_validation_init')
    gcdt_signals.config_validation_finalized.send((context, config))
    if context['command'] in \
            DEFAULT_CONFIG.get(context['tool'], {}).get('non_config_commands', []):
        pass  # we do not require a config for this command
    elif tool not in config and tool != 'gcdt':
        context['error'] = 'Configuration missing for \'%s\'.' % tool
        log.error(context['error'])
        gcdt_signals.error.send((context, config))
        return 1
    log.debug('### config_validation_finalized')

    ## check credentials are valid (AWS services)
    # DEPRECATED, use gcdt-logon plugin instead
    if are_credentials_still_valid(awsclient):
        context['error'] = \
            'Your credentials have expired... Please renew and try again!'
        log.error(context['error'])
        gcdt_signals.error.send((context, config))
        return 1

    ## bundle step
    gcdt_signals.bundle_pre.send((context, config))
    log.debug('### bundle_pre')
    gcdt_signals.bundle_init.send((context, config))
    log.debug('### bundle_init')
    gcdt_signals.bundle_finalized.send((context, config))
    log.debug('### bundle_finalized')
    if 'error' in context:
        log.error(context['error'])
        gcdt_signals.error.send((context, config))
        return 1

    ## dispatch command providing context and config (= tooldata)
    gcdt_signals.command_init.send((context, config))
    log.debug('### command_init')
    try:
        if tool == 'gcdt':
            conf = config  # gcdt works on the whole config
        else:
            conf = config.get(tool, {})
        exit_code = cmd.dispatch(arguments,
                                 context=context,
                                 config=conf)
    except GracefulExit:
        raise
    except Exception as e:
        log.debug(traceback.format_exc())
        context['error'] = str(e)
        log.error(context['error'])
        exit_code = 1
    if exit_code:
        if 'error' not in context or context['error'] == '':
            context['error'] = '\'%s\' command failed with exit code 1' % command
        gcdt_signals.error.send((context, config))
        return 1

    gcdt_signals.command_finalized.send((context, config))
    log.debug('### command_finalized')

    # TODO reporting (in case you want to get a summary / output to the user)

    gcdt_signals.finalized.send(context)
    log.debug('### finalized')
    return 0