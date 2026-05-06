def do_ssh(self,args):
        """SSH to an instance. ssh -h for detailed help"""
        parser = CommandArgumentParser("ssh")
        parser.add_argument(dest='instance',help='instance index or name');
        parser.add_argument('-a','--address-number',default='0',dest='interface-number',help='instance id of the instance to ssh to');
        parser.add_argument('-ii','--ignore-host-key',dest='ignore-host-key',default=False,action='store_true',help='Ignore host key')
        parser.add_argument('-ne','--no-echo',dest='no-echo',default=False,action='store_true',help='Do not echo command')
        parser.add_argument('-L',dest='forwarding',nargs='*',help="port forwarding string of the form: {localport}:{host-visible-to-instance}:{remoteport} or {port}")
        parser.add_argument('-R','--replace-key',dest='replaceKey',default=False,action='store_true',help="Replace the host's key. This is useful when AWS recycles an IP address you've seen before.")
        parser.add_argument('-Y','--keyscan',dest='keyscan',default=False,action='store_true',help="Perform a keyscan to avoid having to say 'yes' for a new host. Implies -R.")
        parser.add_argument('-B','--background',dest='background',default=False,action='store_true',help="Run in the background. (e.g., forward an ssh session and then do other stuff in aws-shell).")
        parser.add_argument('-v',dest='verbosity',default=0,action=VAction,nargs='?',help='Verbosity. The more instances, the more verbose');        
        parser.add_argument('-m',dest='macro',default=False,action='store_true',help='{command} is a series of macros to execute, not the actual command to run on the host');
        parser.add_argument(dest='command',nargs='*',help="Command to run on all hosts.") # consider adding a filter option later
        args = vars(parser.parse_args(args))

        interfaceNumber = int(args['interface-number'])
        forwarding = args['forwarding']
        replaceKey = args['replaceKey']
        keyscan = args['keyscan']
        background = args['background']
        verbosity = args['verbosity']
        ignoreHostKey = args['ignore-host-key']
        noEcho = args['no-echo']

        # Figure out the host to connect to:
        target = args['instance']
        try:
            index = int(args['instance'])
            instances = self.scalingGroupDescription['AutoScalingGroups'][0]['Instances']
            instance = instances[index]
            target = instance['InstanceId']
        except ValueError: # if args['instance'] is not an int, for example.
            pass
        
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
            
        ssh(target,interfaceNumber,forwarding,replaceKey,keyscan,background,verbosity,command,ignoreHostKey=ignoreHostKey,echoCommand = not noEcho)