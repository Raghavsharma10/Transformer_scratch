def run(self, context=None, options=None):
        """
        Run axe against the current page.

        :param context: which page part(s) to analyze and/or what to exclude.
        :param options: dictionary of aXe options.
        """
        template = (
            "var callback = arguments[arguments.length - 1];"
            + "axe.run(%s).then(results => callback(results))"
        )
        args = ""

        # If context parameter is passed, add to args
        if context is not None:
            args += "%r" % context
        # Add comma delimiter only if both parameters are passed
        if context is not None and options is not None:
            args += ","
        # If options parameter is passed, add to args
        if options is not None:
            args += "%s" % options

        command = template % args
        response = self.selenium.execute_async_script(command)
        return response