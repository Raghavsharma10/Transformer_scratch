def run(self, check_interval=300):
        """ Run the daemon

        :type check_interval: int
        :param check_interval: Delay in seconds between checks
        """
        while True:
            # Read configuration from the config file if present, else fall
            # back to command line options
            if args.config:
                config = config_file_parser.get_configuration(args.config)
                access_key_id = config['access-key-id']
                secret_access_key = config['secret-access-key']
                region = config['region']
            else:
                access_key_id = args.access_key_id
                secret_access_key = args.secret_access_key
                region = args.region

            # Connect to AWS
            connection = connection_manager.connect_to_ec2(
                region, access_key_id, secret_access_key)

            snapshot_manager.run(connection)

            logger.info('Sleeping {} seconds until next check'.format(
                check_interval))
            time.sleep(check_interval)