def do_browse(self,args):
        """Open the current stack in a browser."""
        rawStack = self.wrappedStack['rawStack']
        os.system("open -a \"Google Chrome\" https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stack/detail?stackId={}".format(rawStack.stack_id))