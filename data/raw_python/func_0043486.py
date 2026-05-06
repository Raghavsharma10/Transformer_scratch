def main():
    """
    Entry point for the CLI interface
    """
   
    argparser = argparse.ArgumentParser(
            description='Blockstack-file version {}'.format(__version__))

    subparsers = argparser.add_subparsers(
            dest='action', help='The file command to take [get/put/delete]')

    parser = subparsers.add_parser(
            'init',
            help='Initialize this host to start sending and receiving files')
    parser.add_argument(
            '--config', action='store',
            help='path to the config file to use (default is %s)' % CONFIG_PATH)
    parser.add_argument(
            '--blockchain_id', action='store',
            help='the recipient blockchain ID to use'),
    parser.add_argument(
            '--hostname', action='store',
            help='the recipient hostname to use')

    parser = subparsers.add_parser(
            'reset',
            help='Reset this host\'s key')
    parser.add_argument(
            '--config', action='store',
            help='path to the config file to use (default is %s)' % CONFIG_PATH)
    parser.add_argument(
            '--blockchain_id', action='store',
            help='the recipient blockchain ID to use'),
    parser.add_argument(
            '--hostname', action='store',
            help='the recipient hostname to use')

    parser = subparsers.add_parser(
            'get',
            help='Get a file')
    parser.add_argument(
            '--config', action='store',
            help='path to the config file to use (default is %s)' % CONFIG_PATH)
    parser.add_argument(
            '--blockchain_id', action='store',
            help='the recipient blockchain ID to use'),
    parser.add_argument(
            '--hostname', action='store',
            help='the recipient hostname to use')
    parser.add_argument(
            '--passphrase', action='store',
            help='decryption passphrase')
    parser.add_argument(
            '--wallet', action='store',
            help='path to your Blockstack wallet')
    parser.add_argument(
            'sender_blockchain_id', action='store',
            help='the sender\'s blockchain ID')
    parser.add_argument(
            'data_name', action='store',
            help='Public name of the file to fetch')
    parser.add_argument(
            'output_path', action='store', nargs='?',
            help='[optional] destination path to save the file; defaults to stdout')

    parser = subparsers.add_parser(
            'put',
            help='Share a file')
    parser.add_argument(
            '--config', action='store',
            help='path to the config file to use (default is %s)' % CONFIG_PATH)
    parser.add_argument(
            '--blockchain_id', action='store',
            help='the sender blockchain ID to use'),
    parser.add_argument(
            '--hostname', action='store',
            help='the sender hostname to use')
    parser.add_argument(
            '--passphrase', action='store',
            help='encryption passphrase')
    parser.add_argument(
            '--wallet', action='store',
            help='path to your Blockstack wallet')
    parser.add_argument(
            'input_path', action='store',
            help='Path to the file to share')
    parser.add_argument(
            'data_name', action='store',
            help='Public name of the file to store')
    # recipients come afterwards


    parser = subparsers.add_parser(
            'delete',
            help='Delete a shared file')
    parser.add_argument(
            '--config', action='store',
            help='path to the config file to use (default is %s)' % CONFIG_PATH)
    parser.add_argument(
            '--blockchain_id', action='store',
            help='the sender blockchain ID to use'),
    parser.add_argument(
            '--hostname', action='store',
            help='the sender hostname to use')
    parser.add_argument(
            '--wallet', action='store',
            help='path to your Blockstack wallet')
    parser.add_argument(
            'data_name', action='store',
            help='Public name of the file to delete')

    args, unparsed = argparser.parse_known_args()

    # load up config
    config_path = args.config
    if config_path is None:
        config_path = CONFIG_PATH

    conf = get_config( config_path )
    config_dir = os.path.dirname(config_path)
    blockchain_id = getattr(args, "blockchain_id", None)
    hostname = getattr(args, "hostname", None)
    passphrase = getattr(args, "passphrase", None)
    data_name = getattr(args, "data_name", None)
    wallet_path = getattr(args, "wallet", None)

    if blockchain_id is None:
        blockchain_id = conf['blockchain_id']

    if hostname is None:
        hostname = conf['hostname']

    if wallet_path is None:
        wallet_path = conf['wallet']
    
    if wallet_path is None and config_dir is not None:
        wallet_path = os.path.join(config_dir, blockstack_client.config.WALLET_FILENAME)

    # load wallet 
    if wallet_path is not None and os.path.exists( wallet_path ):
        # load from disk
        log.debug("Load wallet from %s" % wallet_path)
        wallet = blockstack_client.load_wallet( config_dir=config_dir, wallet_path=wallet_path, include_private=True )
        if 'error' in wallet:
            print >> sys.stderr, json.dumps(wallet, sort_keys=True, indent=4 )
            sys.exit(1)

        else:
            wallet = wallet['wallet']

    else:
        # load from RPC
        log.debug("Load wallet from RPC")
        wallet = blockstack_client.dump_wallet(config_path=config_path)
        if 'error' in wallet:
            print >> sys.stderr, json.dumps(wallet, sort_keys=True, indent=4)
            sys.exit(1)

    log.debug("Process %s" %  args.action)
    if args.action in ['init', 'reset']:
        # (re)key
        res = file_key_regenerate( blockchain_id, hostname, config_path=config_path, wallet_keys=wallet ) 
        if 'error' in res:
            print >> sys.stderr, json.dumps(res, sort_keys=True, indent=4 )
            sys.exit(1)
        

    if args.action == 'get':
        # get a file
        sender_blockchain_id = args.sender_blockchain_id
        output_path = args.output_path

        tmp = False
        if output_path is None:
            fd, path = tempfile.mkstemp( prefix='blockstack-file-', dir=config_dir )
            os.close(fd)
            output_path = path
            tmp = True

        res = file_get( blockchain_id, hostname, sender_blockchain_id, data_name, output_path, passphrase=passphrase, config_path=config_path, wallet_keys=wallet )
        if 'error' in res:
            print >> sys.stderr, json.dumps(res, sort_keys=True, indent=4 )
            sys.exit(1)

        if tmp:
            # print to stdout 
            with open(output_path, "r") as f:
                while True:
                    buf = f.read(65536)
                    if len(buf) == 0:
                        break

                    sys.stdout.write(buf)

            os.unlink(output_path)

    elif args.action == 'put':
        # put a file
        recipients = unparsed
        input_path = args.input_path
        res = file_put( blockchain_id, hostname, recipients, data_name, input_path, passphrase=passphrase, config_path=config_path, wallet_keys=wallet )
        if 'error' in res:
            print >> sys.stderr, json.dumps(res, sort_keys=True, indent=4 )
            sys.exit(1)

    elif args.action == 'delete':
        # delete a file
        res = file_delete( blockchain_id, data_name, config_path=config_path, wallet_keys=wallet )
        if 'error' in res:
            print >> sys.stderr, json.dumps(res, sort_keys=True, indent=4 )
            sys.exit(1)

    
    print >> sys.stderr, json.dumps({'status': True}, sort_keys=True, indent=4 )
    sys.exit(0)