def restore_iptables(firewall_rules):
    """Restores and saves firewall rules from the firewall_rules file

    :param firewall_rules: (str) Full path to the firewall rules file
    :return: None
    :raises OSError
    """
    log = logging.getLogger(mod_logger + '.restore_iptables')
    log.info('Restoring firewall rules from file: {f}'.format(f=firewall_rules))

    # Ensure the firewall rules file exists
    if not os.path.isfile(firewall_rules):
        msg = 'Unable to restore iptables, file not found: {f}'.format(f=firewall_rules)
        log.error(msg)
        raise OSError(msg)

    # Restore the firewall rules
    log.info('Restoring iptables from file: {f}'.format(f=firewall_rules))
    command = ['/sbin/iptables-restore', firewall_rules]
    try:
        result = run_command(command)
    except CommandError:
        _, ex, trace = sys.exc_info()
        msg = 'Unable to restore firewall rules from file: {f}\n{e}'.format(f=firewall_rules, e=str(ex))
        log.error(msg)
        raise OSError(msg)
    log.info('Restoring iptables produced output:\n{o}'.format(o=result['output']))

    # Save iptables
    log.info('Saving iptables...')
    command = ['/etc/init.d/iptables', 'save']
    try:
        result = run_command(command)
    except CommandError:
        _, ex, trace = sys.exc_info()
        msg = 'Unable to save firewall rules\n{e}'.format(e=str(ex))
        log.error(msg)
        raise OSError(msg)
    log.info('Saving iptables produced output:\n{o}'.format(o=result['output']))