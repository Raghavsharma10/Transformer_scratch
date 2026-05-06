def updateBeforeDecorator(function):
        """ Function updateAfterDecorator
        Decorator to ensure local dict is sync with remote foreman
        """
        def _updateBeforeDecorator(self, *args, **kwargs):
            if self.forceFullSync:
                self.reload()
            return function(self, *args, **kwargs)
        return _updateBeforeDecorator