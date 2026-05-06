def convert(self, *args, **kwargs):
        """
        Yes it is, thanks captain.
        """

        self.strings()
        self.metadata()

        # save file
        self.result.save(self.output())