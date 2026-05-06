def parse(self, subscription):
        """ Fetch the function registered for a certain subscription """

        for name in self.methods:
            tag = bytes(name.encode('utf-8'))
            if subscription.startswith(tag):
                fun = self.methods.get(name)
                message = subscription[len(tag):]
                return tag, message, fun
        return None, None, None