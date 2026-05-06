def do_stopInstance(self,args):
        """Stop specified instance"""
        parser = CommandArgumentParser("stopInstance")
        parser.add_argument(dest='instance',help='instance index or name');
        parser.add_argument('-f','--force',action='store_true',dest='force',help='instance index or name');
        args = vars(parser.parse_args(args))

        instanceId = args['instance']
        force = args['force']
        try:
            index = int(instanceId)
            instances = self.scalingGroupDescription['AutoScalingGroups'][0]['Instances']
            instanceId = instances[index]
        except ValueError:
            pass

        client = AwsConnectionFactory.getEc2Client()
        client.stop_instances(InstanceIds=[instanceId['InstanceId']],Force=force)