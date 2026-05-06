def do_printActivity(self,args):
        """Print scaling activity details"""
        parser = CommandArgumentParser("printActivity")
        parser.add_argument(dest='index',type=int,help='refresh');
        args = vars(parser.parse_args(args))
        index = args['index']

        activity = self.activities[index]
        pprint(activity)