def parse_args(argv):
    """
    Use Argparse to parse command-line arguments.

    :param argv: list of arguments to parse (``sys.argv[1:]``)
    :type argv: :std:term:`list`
    :return: parsed arguments
    :rtype: :py:class:`argparse.Namespace`
    """
    p = argparse.ArgumentParser(
        description='webhook2lambda2sqs - Generate code and manage '
                    'infrastructure for receiving webhooks with AWS API '
                    'Gateway and pushing to SQS via Lambda - <%s>' % PROJECT_URL
    )
    p.add_argument('-c', '--config', dest='config', type=str,
                   action='store', default='config.json',
                   help='path to config.json (default: ./config.json)')
    p.add_argument('-v', '--verbose', dest='verbose', action='count',
                   default=0,
                   help='verbose output. specify twice for debug-level output.')
    p.add_argument('-V', '--version', action='version',
                   version='webhook2lambda2sqs v%s <%s>' % (
                       VERSION, PROJECT_URL
                   ))
    p.add_argument('-T', '--tf-version', dest='tf_ver', action='store',
                   type=str, default='0.9.0',
                   help='terraform version to generate configurations for')
    subparsers = p.add_subparsers(title='Action (Subcommand)', dest='action',
                                  metavar='ACTION', description='Action to '
                                  'perform; each action may take further '
                                  'parameters. Use ACTION -h for subcommand-'
                                  'specific options and arguments.')
    subparsers.add_parser(
        'generate', help='generate lambda function and terraform configs in ./'
    )
    tf_parsers = [
        ('genapply', 'generate function and terraform configs in ./, then run '
                     'terraform apply'),
        ('plan', 'run terraform plan to show changes which will be made'),
        ('apply', 'run terraform apply to apply changes/create infrastructure'),
        ('destroy',
         'run terraform destroy to completely destroy infrastructure')
    ]
    tf_p_objs = {}
    for cname, chelp in tf_parsers:
        tf_p_objs[cname] = subparsers.add_parser(cname, help=chelp)
        tf_p_objs[cname].add_argument('-t', '--terraform-path', dest='tf_path',
                                      action='store', default='terraform',
                                      type=str, help='path to terraform '
                                                     'binary, if not in PATH')
        tf_p_objs[cname].add_argument('-S', '--no-stream-tf', dest='stream_tf',
                                      action='store_false', default=True,
                                      help='DO NOT stream Terraform output to '
                                           'STDOUT (combined) in realtime')
    apilogparser = subparsers.add_parser('apilogs', help='show last 10 '
                                         'CloudWatch Logs entries for the '
                                         'API Gateway')
    apilogparser.add_argument('-c', '--count', dest='log_count', type=int,
                              default=10, help='number of log entries to show '
                              '(default 10')
    logparser = subparsers.add_parser('logs', help='show last 10 CloudWatch '
                                      'Logs entries for the function')
    logparser.add_argument('-c', '--count', dest='log_count', type=int,
                           default=10, help='number of log entries to show '
                                            '(default 10')
    queueparser = subparsers.add_parser('queuepeek', help='show messages from '
                                        'one or all of the SQS queues')
    queueparser.add_argument('-n', '--name', type=str, dest='queue_name',
                             default=None, help='queue name to read (defaults '
                                                'to None to read all)')
    queueparser.add_argument('-d', '--delete', action='store_true',
                             dest='queue_delete', default=False,
                             help='delete messages after reading')
    queueparser.add_argument('-c', '--count', dest='msg_count', type=int,
                             default=10, help='number of messages to read from '
                                              'each queue (default 10)')
    testparser = subparsers.add_parser('test', help='send test message to '
                                                    'one or more endpoints')
    testparser.add_argument('-t', '--terraform-path', dest='tf_path',
                            action='store', default='terraform',
                            type=str, help='path to terraform '
                            'binary, if not in PATH')
    testparser.add_argument('-n', '--endpoint-name', dest='endpoint_name',
                            type=str, default=None,
                            help='endpoint name (default: None, to send to '
                                 'all endpoints)')
    subparsers.add_parser(
        'example-config', help='write example config to STDOUT and description '
                               'of it to STDERR, then exit'
    )
    args = p.parse_args(argv)
    if args.action is None:
        # for py3, which doesn't raise on this
        sys.stderr.write("ERROR: too few arguments\n")
        raise SystemExit(2)
    return args