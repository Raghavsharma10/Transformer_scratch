def register(self, callback, name):
        'Register a callback on server and on connected clients.'
        server.CALLBACKS[name] = callback
        self.run('''
            window.skink.%s = function(args=[]) {
                window.skink.call("%s", args);
            }''' % (name, name))