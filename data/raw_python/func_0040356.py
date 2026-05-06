def do_stack(self,args):
        """Go to the specified stack. stack -h for detailed help"""
        parser = CommandArgumentParser("stack")
        parser.add_argument(dest='stack',help='stack index or name');
        parser.add_argument('-a','--asg',dest='asg',help='descend into specified asg');
        args = vars(parser.parse_args(args))

        try:
            index = int(args['stack'])
            if self.stackList == None:
                self.do_stacks('-s')
            stack = AwsConnectionFactory.instance.getCfResource().Stack(self.stackList[index]['StackName'])
        except ValueError:
            stack = AwsConnectionFactory.instance.getCfResource().Stack(args['stack'])


        if 'asg' in args:
            AwsProcessor.processorFactory.Stack(stack,stack.name,self).onecmd('asg {}'.format(args['asg']))
        AwsProcessor.processorFactory.Stack(stack,stack.name,self).cmdloop()