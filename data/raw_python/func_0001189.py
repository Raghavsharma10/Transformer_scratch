def transform(function):
        """Return a processor for a style's "transform" function.
        """
        def transform_fn(_, result):
            if isinstance(result, Nothing):
                return result

            lgr.debug("Transforming %r with %r", result, function)
            try:
                return function(result)
            except:
                exctype, value, tb = sys.exc_info()
                try:
                    new_exc = StyleFunctionError(function, exctype, value)
                    # Remove the "During handling ..." since we're
                    # reraising with the traceback.
                    new_exc.__cause__ = None
                    six.reraise(StyleFunctionError, new_exc, tb)
                finally:
                    # Remove circular reference.
                    # https://docs.python.org/2/library/sys.html#sys.exc_info
                    del tb
        return transform_fn