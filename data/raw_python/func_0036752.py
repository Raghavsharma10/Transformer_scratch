def add_callback(self, event, cb, args=None):
        '''Add a callback to this node.

        Callbacks are called when the specified event occurs. The available
        events depends on the specific node type. Args should be a value to
        pass to the callback when it is called. The callback should be of the
        format:

        def callback(node, value, cb_args):

        where node will be the node that called the function, value is the
        relevant information for the event, and cb_args are the arguments you
        registered with the callback.

        '''
        if event not in self._cbs:
            raise exceptions.NoSuchEventError
        self._cbs[event] = [(cb, args)]