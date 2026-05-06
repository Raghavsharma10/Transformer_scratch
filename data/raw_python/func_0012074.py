def execute(self, context):
        """Execute the strategies on the given context"""
        for ware in self.middleware:
            ware.premessage(context)
            context = ware.bind(context)
            ware.postmessage(context)
        return context