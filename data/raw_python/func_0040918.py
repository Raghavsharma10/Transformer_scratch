def do_run(self,args):
        """SSH to each instance in turn and run specified command"""
        parser = CommandArgumentParser("run")
        parser.add_argument('-R','--replace-key',dest='replaceKey',default=False,action='store_true',help="Replace the host's key. This is useful when AWS recycles an IP address you've seen before.")
        parser.add_argument('-Y','--keyscan',dest='keyscan',default=False,action='store_true',help="Perform a keyscan to avoid having to say 'yes' for a new host. Implies -R.")
        parser.add_argument('-ii','--ignore-host-key',dest='ignore-host-key',default=False,action='store_true',help='Ignore host key')
        parser.add_argument('-ne','--no-echo',dest='no-echo',default=False,action='store_true',help='Do not echo command')
        parser.add_argument(dest='command',nargs='+',help="Command to run on all hosts.") # consider adding a filter option later
        parser.add_argument('-v',dest='verbosity',default=0,action=VAction,nargs='?',help='Verbosity. The more instances, the more verbose');        
        parser.add_argument('-j',dest='jobs',type=int,default=1,help='Number of hosts to contact in parallel');
        parser.add_argument('-s',dest='skip',type=int,default=0,help='Skip this many hosts');
        parser.add_argument('-m',dest='macro',default=False,action='store_true',help='{command} is a series of macros to execute, not the actual command to run on the host');
        args = vars(parser.parse_args(args))

        replaceKey = args['replaceKey']
        keyscan = args['keyscan']
        verbosity = args['verbosity']
        jobs = args['jobs']
        skip = args['skip']
        ignoreHostKey = args['ignore-host-key']
        noEcho = args['no-echo']

        instances = self.scalingGroupDescription['AutoScalingGroups'][0]['Instances']
        instances = instances[skip:]
        # if replaceKey or keyscan:
        #     for instance in instances:
        #         stdplus.resetKnownHost(instance)

        if args['macro']:
            if len(args['command']) > 1:
                print("Only one macro may be specified with the -m switch.")
                return
            else:
                macro = args['command'][0]
                print("Macro:{}".format(macro))
                command = Config.config['ssh-macros'][macro]
        else:
            command = ' '.join(args['command'])
            
        Parallel(n_jobs=jobs)(
            delayed(ssh)(instance['InstanceId'],0,[],replaceKey,keyscan,False,verbosity,command,ignoreHostKey=ignoreHostKey,echoCommand=not noEcho,name="{}:{}: ".format(instance['index'],instance['InstanceId'])) for instance in instances
        )