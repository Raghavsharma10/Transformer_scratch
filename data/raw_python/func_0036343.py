def iterate(self, func, args=None, filter=[]):
        '''Call a function on the root node, and recursively all its children.

        This is a depth-first iteration.

        @param func The function to call. Its declaration must be
                    'def blag(node, args)', where 'node' is the current node
                    in the iteration and args is the value of @ref args.
        @param args Extra arguments to pass to the function at each iteration.
                    Pass multiple arguments in as a tuple.
        @param filter A list of filters to apply before calling func for each
                      node in the iteration. If the filter is not True,
                      @ref func will not be called for that node. Each filter
                      entry should be a string, representing on of the is_*
                      properties (is_component, etc), or a function object.
        @return The results of the calls to @ref func in a list.

        '''
        return self._root.iterate(func, args, filter)