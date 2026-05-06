def execute(self, method, args, ref):
        """ Execute the method with args """

        response = {'result': None, 'error': None, 'ref': ref}
        fun = self.methods.get(method)
        if not fun:
            response['error'] = 'Method `{}` not found'.format(method)
        else:
            try:
                response['result'] = fun(*args)
            except Exception as exception:
                logging.error(exception, exc_info=1)
                response['error'] = str(exception)
        return response