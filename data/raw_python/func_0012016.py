def interface(self, context):
        """Implement the interface for the adapter object"""
        self.context = context
        self.callback = self.context.get("callback")