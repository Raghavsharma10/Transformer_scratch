def rewrite(self, path, expand, newmethod = None, host = None, vhost = None, method = [b'GET', b'HEAD'], keepquery = True):
        "Automatically rewrite a request to another location"
        async def func(env):
            newpath = self.expand(env.path_match, expand)
            if keepquery and getattr(env, 'querystring', None):
                if b'?' in newpath:
                    newpath += b'&' + env.querystring
                else:
                    newpath += b'?' + env.querystring
            await env.rewrite(newpath, newmethod)
        self.route(path, func)