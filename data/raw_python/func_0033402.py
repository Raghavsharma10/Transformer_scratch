def send_example(self,
                     *args,
                     **kwargs
                     ):
        """Send a labeled or unlabeled example to the VW instance.
        If 'parse_result' kwarg is False, ignore the result and return None.

        All other parameters are passed to self.send_line().

        Returns a VWResult object.
        """
        # Pop out the keyword argument 'parse_result' if given
        parse_result = kwargs.pop('parse_result', True)
        line = self.make_line(*args, **kwargs)
        result = self.send_line(line, parse_result=parse_result)
        return result