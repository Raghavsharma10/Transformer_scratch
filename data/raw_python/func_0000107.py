def updateAfterDecorator(function):
        """ Function updateAfterDecorator
        Decorator to ensure local dict is sync with remote foreman
        """
        def _updateAfterDecorator(self, *args, **kwargs):
            ret = function(self, *args, **kwargs)
            self.reload()
            return ret
        return _updateAfterDecorator