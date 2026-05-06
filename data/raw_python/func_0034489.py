def main():  # pylint: disable-msg=R0912,R0915
    """Main."""
    parser = optparse.OptionParser()
    parser.usage = textwrap.dedent("""\
    %prog {--run|--install_key|--dump_config} [options]

    SSH command authenticator.

    Used to restrict which commands can be run via trusted SSH keys.
    """)

    group = optparse.OptionGroup(
        parser, 'Run Mode Options',
        'These options determine in which mode the authprogs '
        'program runs.')
    group.add_option(
        '-r', '--run', dest='run', action='store_true',
        help='Act as ssh command authenticator. Use this '
        'when calling from authorized_keys.')
    group.add_option(
        '--dump_config', dest='dump_config',
        action='store_true',
        help='Dump configuration (python format) '
        'to standard out and exit.')
    group.add_option(
        '--install_key', dest='install_key',
        help='Install the named ssh public key file to '
        'authorized_keys.', metavar='FILE')
    parser.add_option_group(group)

    group = optparse.OptionGroup(parser, 'Other Options')
    group.add_option(
        '--keyname', dest='keyname',
        help='Name for this key, used when matching '
        'config blocks.')
    group.add_option(
        '--configfile', dest='configfile',
        help='Path to authprogs configuration file. '
        'Defaults to ~/.ssh/authprogs.yaml',
        metavar='FILE')
    group.add_option(
        '--configdir', dest='configdir',
        help='Path to authprogs configuration directory. '
        'Defaults to ~/.ssh/authprogs.d',
        metavar='DIR')
    group.add_option('--logfile', dest='logfile',
                     help='Write logging info to this file. '
                     'Defaults to no logging.',
                     metavar='FILE')
    group.add_option('--debug', dest='debug', action='store_true',
                     help='Write additional debugging information '
                     'to --logfile')
    group.add_option('--authorized_keys', dest='authorized_keys',
                     default=os.path.expanduser('~/.ssh/authorized_keys'),
                     help='Location of authorized_keys file for '
                     '--install_key. Defaults to ~/.ssh/authorized_keys',
                     metavar='FILE')
    parser.add_option_group(group)

    opts, args = parser.parse_args()
    if args:
        sys.exit('authprogs does not accept commandline arguments.')

    if not opts.configfile:
        cfg = os.path.expanduser('~/.ssh/authprogs.yaml')
        if os.path.isfile(cfg):
            opts.configfile = cfg
    if not opts.configdir:
        cfg = os.path.expanduser('~/.ssh/authprogs.d')
        if os.path.isdir(cfg):
            opts.configdir = cfg

    if opts.debug and not opts.logfile:
        parser.error('--debug requires use of --logfile')

    ap = None
    try:
        ap = AuthProgs(logfile=opts.logfile,  # pylint: disable-msg=C0103
                       configfile=opts.configfile,
                       configdir=opts.configdir,
                       debug=opts.debug,
                       keyname=opts.keyname)

        if opts.dump_config:
            ap.dump_config()
            sys.exit(0)

        elif opts.install_key:
            try:
                ap.install_key(opts.install_key, opts.authorized_keys)
                sys.stderr.write('Key installed successfully.\n')
                sys.exit(0)
            except InstallError as err:
                sys.stderr.write('Key install failed: %s' % err)
                sys.exit(1)

        elif opts.run:
            ap.exec_command()
            sys.exit('authprogs command returned - should '
                     'never happen.')
        else:
            parser.error('Not sure what to do. Consider --help')

    except SSHEnvironmentError as err:
        ap.log('SSHEnvironmentError "%s"\n%s\n' % (
               err, traceback.format_exc()))
        sys.exit('authprogs: %s' % err)
    except ConfigError as err:
        ap.log('ConfigError "%s"\n%s\n' % (
               err, traceback.format_exc()))
        sys.exit('authprogs: %s' % err)
    except CommandRejected as err:
        sys.exit('authprogs: %s' % err)
    except Exception as err:
        if ap:
            ap.log('Unexpected exception: %s\n%s\n' % (
                   err, traceback.format_exc()))
        else:
            sys.stderr.write('Unexpected exception: %s\n%s\n' % (
                             err, traceback.format_exc()))
        sys.exit('authprogs experienced an unexpected exception.')