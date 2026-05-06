def do_printPolicy(self,args):
        """Print the autoscaling policy"""
        parser = CommandArgumentParser("printPolicy")
        args = vars(parser.parse_args(args))

        policy = self.client.describe_policies(AutoScalingGroupName=self.scalingGroup)
        pprint(policy)