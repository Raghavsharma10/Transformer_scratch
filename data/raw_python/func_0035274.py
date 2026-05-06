def prewalk_hook(self, dispatcher, node):
        """
        This is for the Unparser to use as a prewalk hook.
        """

        self.walk(dispatcher, node)
        self.finalize()
        return node