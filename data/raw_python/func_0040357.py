def do_stack_resource(self, args):
        """Use specified stack resource. stack_resource -h for detailed help."""
        parser = CommandArgumentParser()
        parser.add_argument('-s','--stack-name',dest='stack-name',help='name of the stack resource');
        parser.add_argument('-i','--logical-id',dest='logical-id',help='logical id of the child resource');
        args = vars(parser.parse_args(args))

        stackName = args['stack-name']
        logicalId = args['logical-id']

        self.stackResource(stackName,logicalId)