def main(arv=None):
    """lambda-uploader command line interface."""
    # Check for Python 2.7 or later
    if sys.version_info[0] < 3 and not sys.version_info[1] == 7:
        raise RuntimeError('lambda-uploader requires Python 2.7 or later')

    import argparse

    parser = argparse.ArgumentParser(
            description='Simple way to create and upload python lambda jobs')

    parser.add_argument('--version', '-v', action='version',
                        version=lambda_uploader.__version__)
    parser.add_argument('--no-upload', dest='no_upload',
                        action='store_const', help='dont upload the zipfile',
                        const=True)
    parser.add_argument('--no-clean', dest='no_clean',
                        action='store_const',
                        help='dont cleanup the temporary workspace',
                        const=True)
    parser.add_argument('--publish', '-p', dest='publish',
                        action='store_const',
                        help='publish an upload to an immutable version',
                        const=True)
    parser.add_argument('--virtualenv', '-e',
                        help='use specified virtualenv instead of making one',
                        default=None)
    parser.add_argument('--extra-files', '-x',
                        action='append',
                        help='include file or directory path in package',
                        default=[])
    parser.add_argument('--no-virtualenv', dest='no_virtualenv',
                        action='store_const',
                        help='do not create or include a virtualenv at all',
                        const=True)
    parser.add_argument('--role', dest='role',
                        default=getenv('LAMBDA_UPLOADER_ROLE'),
                        help=('IAM role to assign the lambda function, '
                              'can be set with $LAMBDA_UPLOADER_ROLE'))
    parser.add_argument('--variables', dest='variables',
                        help='add environment variables')
    parser.add_argument('--profile', dest='profile',
                        help='specify AWS cli profile')
    parser.add_argument('--requirements', '-r', dest='requirements',
                        help='specify a requirements.txt file')
    alias_help = 'alias for published version (WILL SET THE PUBLISH FLAG)'
    parser.add_argument('--alias', '-a', dest='alias',
                        default=None, help=alias_help)
    parser.add_argument('--alias-description', '-m', dest='alias_description',
                        default=None, help='alias description')
    parser.add_argument('--s3-bucket', '-s', dest='s3_bucket',
                        help='S3 bucket to store the lambda function in',
                        default=None)
    parser.add_argument('--s3-key', '-k', dest='s3_key',
                        help='Key name of the lambda function s3 object',
                        default=None)
    parser.add_argument('--config', '-c', help='Overrides lambda.json',
                        default='lambda.json')
    parser.add_argument('function_dir', default=getcwd(), nargs='?',
                        help='lambda function directory')
    parser.add_argument('--no-build', dest='no_build',
                        action='store_const', help='dont build the sourcecode',
                        const=True)

    verbose = parser.add_mutually_exclusive_group()
    verbose.add_argument('-V', dest='loglevel', action='store_const',
                         const=logging.INFO,
                         help="Set log-level to INFO.")
    verbose.add_argument('-VV', dest='loglevel', action='store_const',
                         const=logging.DEBUG,
                         help="Set log-level to DEBUG.")
    parser.set_defaults(loglevel=logging.WARNING)

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    try:
        _execute(args)
    except Exception:
        print(TRACEBACK_MESSAGE
              % (INTERROBANG, lambda_uploader.__version__,
                 boto3_version, botocore_version),
              file=sys.stderr)

        traceback.print_exc()
        sys.stderr.flush()
        sys.exit(1)