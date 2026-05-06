def main(args=None):
    """
    Main entry point
    """
    # parse args
    if args is None:
        args = parse_args(sys.argv[1:])

    # dump example config if that action
    if args.action == 'example-config':
        conf, doc = Config.example_config()
        print(conf)
        sys.stderr.write(doc + "\n")
        return

    # set logging level
    if args.verbose > 1:
        set_log_debug()
    elif args.verbose == 1:
        set_log_info()

    # get our config
    config = Config(args.config)

    if args.action == 'logs':
        aws = AWSInfo(config)
        aws.show_cloudwatch_logs(count=args.log_count)
        return

    if args.action == 'apilogs':
        api_id = get_api_id(config, args)
        aws = AWSInfo(config)
        aws.show_cloudwatch_logs(
            count=args.log_count,
            grp_name='API-Gateway-Execution-Logs_%s/%s' % (
                api_id, config.stage_name
            )
        )
        return

    if args.action == 'queuepeek':
        aws = AWSInfo(config)
        aws.show_queue(name=args.queue_name, delete=args.queue_delete,
                       count=args.msg_count)
        return

    if args.action == 'test':
        run_test(config, args)
        return

    if args.action in ['apply', 'genapply', 'plan', 'destroy']:
        runner = TerraformRunner(config, args.tf_path)
        tf_ver = runner.tf_version
    else:
        tf_ver = tuple(
            [int(x) for x in args.tf_ver.split('.')]
        )

    # if generate or genapply, generate the configs
    if args.action == 'generate' or args.action == 'genapply':
        func_gen = LambdaFuncGenerator(config)
        func_src = func_gen.generate()
        # @TODO: also write func_source to disk
        tf_gen = TerraformGenerator(config, tf_ver=tf_ver)
        tf_gen.generate(func_src)

    # if only generate, exit now
    if args.action == 'generate':
        return

    # run the terraform action
    if args.action == 'apply' or args.action == 'genapply':
        runner.apply(args.stream_tf)
        # conditionally set API Gateway Method settings
        if config.get('api_gateway_method_settings') is not None:
            aws = AWSInfo(config)
            aws.set_method_settings()
    elif args.action == 'plan':
        runner.plan(args.stream_tf)
    else:  # destroy
        runner.destroy(args.stream_tf)