def do_copy(self,args):
        """Copy specified id to stack. copy -h for detailed help."""
        parser = CommandArgumentParser("copy")
        parser.add_argument('-a','--asg',dest='asg',nargs='+',required=False,default=[],help='Copy specified ASG info.')
        parser.add_argument('-o','--output',dest='output',nargs='+',required=False,default=[],help='Copy specified output info.')        
        args = vars(parser.parse_args(args))
        values = []
        if args['output']:
            values.extend(self.getOutputs(args['output']))
        if args['asg']:
            for asg in args['asg']:
                try:
                    index = int(asg)
                    asgSummary = self.wrappedStack['resourcesByTypeIndex']['AWS::AutoScaling::AutoScalingGroup'][index]
                except:
                    asgSummary = self.wrappedStack['resourcesByTypeName']['AWS::AutoScaling::AutoScalingGroup'][asg]
                values.append(asgSummary.physical_resource_id)
        print("values:{}".format(values))
        pyperclip.copy("\n".join(values))