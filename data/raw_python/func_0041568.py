def do_refresh(self,args):
        """Refresh view of the current stack. refresh -h for detailed help"""
        self.wrappedStack = self.wrapStack(AwsConnectionFactory.instance.getCfResource().Stack(self.wrappedStack['rawStack'].name))