def do_config(self,args):
        """
        Deal with configuration. Available subcommands:

        * config print - print the current configuration
        * config reload - reload the current configuration from disk
        * config set - change a setting in the configuration
        * config save - save the configuration to disk

        config -h for more details
        """
        parser = CommandArgumentParser("config")
        subparsers = parser.add_subparsers(help='sub-command help',dest='command')
        # subparsers.required=
        subparsers._parser_class = argparse.ArgumentParser # This is to work around `TypeError: __init__() got an unexpected keyword argument 'prog'`
        
        parserPrint = subparsers.add_parser('print',help='Print the current configuration')
        parserPrint.add_argument(dest='keys',nargs='*',help='Key(s) to print')
        
        parserSet = subparsers.add_parser('set',help='Set a configuration value')
        parserSave = subparsers.add_parser('save',help='Save the current configuration')
        parserReload = subparsers.add_parser('reload',help='Reload the configuration from disk')
        args = vars(parser.parse_args(args))

        print("Command:{}".format(args['command']))
        {
            'print' : AwsProcessor.sub_configPrint,
            'set' : AwsProcessor.sub_configSet,
            'save' : AwsProcessor.sub_configSave,
            'reload' : AwsProcessor.sub_configReload
        }[args['command']]( self, args )