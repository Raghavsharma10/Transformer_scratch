def ensure_result(self, supress_exception = False, defaultresult = None):
        '''
        Context manager to ensure returning the result
        '''
        try:
            yield self
        except Exception as exc:
            if not self.done():
                self.set_exception(exc)
            if not supress_exception:
                raise
        except:
            if not self.done():
                self.set_exception(FutureCancelledException('cancelled'))
            raise
        else:
            if not self.done():
                self.set_result(defaultresult)