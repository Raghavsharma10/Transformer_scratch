def cases(self, env, data):
        '''Calls each nested handler until one of them returns nonzero result.

        If any handler returns `None`, it is interpreted as 
        "request does not match, the handler has nothing to do with it and 
        `web.cases` should try to call the next handler".'''
        for handler in self.handlers:
            env._push()
            data._push()
            try:
                result = handler(env, data)
            finally:
                env._pop()
                data._pop()
            if result is not None:
                return result