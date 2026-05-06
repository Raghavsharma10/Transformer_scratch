def do_print(self,args):
        """Print the current stack. print -h for detailed help"""
        parser = CommandArgumentParser("print")
        parser.add_argument('-r','--refresh',dest='refresh',action='store_true',help='refresh view of the current stack')
        parser.add_argument('-i','--include',dest='include',default=None,nargs='+',help='resource types to include')
        parser.add_argument(dest='filters',nargs='*',default=["*"],help='Filter stacks');
        args = vars(parser.parse_args(args))

        if args['refresh']:
            self.do_refresh('')

        self.printStack(self.wrappedStack,args['include'],args['filters'])