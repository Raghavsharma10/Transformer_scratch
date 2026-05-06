def dryRun(self, func, *args, **kwargs):
        """Instead of running function with `*args` and `**kwargs`, just print
           out the function call."""

        print >> self.out, \
              self.formatterDict.get(func, self.defaultFormatter)(func, *args, **kwargs)