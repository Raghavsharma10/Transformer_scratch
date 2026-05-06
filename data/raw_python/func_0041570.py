def do_resource(self,args):
        """Go to the specified resource. resource -h for detailed help"""
        parser = CommandArgumentParser("resource")
        parser.add_argument('-i','--logical-id',dest='logical-id',help='logical id of the child resource');
        args = vars(parser.parse_args(args))

        stackName = self.wrappedStack['rawStack'].name
        logicalId = args['logical-id']
        self.stackResource(stackName,logicalId)