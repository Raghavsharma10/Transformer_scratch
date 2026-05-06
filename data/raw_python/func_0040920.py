def do_startInstance(self,args):
        """Start specified instance"""
        parser = CommandArgumentParser("startInstance")
        parser.add_argument(dest='instance',help='instance index or name');
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
        client.start_instances(InstanceIds=[instanceId['InstanceId']])