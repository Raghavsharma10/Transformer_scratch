def iterate(self, func, args=None, filter=[]):
        '''Call a function on this node, and recursively all its children.

        This is a depth-first iteration.

        @param func The function to call. Its declaration must be
                    'def blag(node, args)', where 'node' is the current node
                    in the iteration and args is the value of @ref args.
        @param args Extra arguments to pass to the function at each iteration.
                    Pass multiple arguments in as a tuple.
        @param filter A list of filters to apply before calling func for each
                      node in the iteration. If the filter is not True,
                      @ref func will not be called for that node. Each filter
                      entry should be a string, representing one of the is_*
                      properties (is_component, etc), or a function object.
        @return The results of the calls to @ref func in a list.

        Example:
        >>> c1 = TreeNode(name='c1')
        >>> c2 = TreeNode(name='c2')
        >>> p = TreeNode(name='p', children={'c1':c1, 'c2':c2})
        >>> c1._parent = p
        >>> c2._parent = p
        >>> def hello(n, args):
        ...     return args[0] + ' ' + n._name
        >>> p.iterate(hello, args=['hello'])
        ['hello p', 'hello c2', 'hello c1']
        >>> p.iterate(hello, args=['hello'], filter=['_name=="c1"'])
        ['hello c1']
        '''
        with self._mutex:
            result = []
            if filter:
                filters_passed = True
                for f in filter:
                    if type(f) == str:
                        if not eval('self.' + f):
                            filters_passed = False
                            break
                    else:
                        if not f(self):
                            filters_passed = False
                            break
                if filters_passed:
                    result = [func(self, args)]
            else:
                result = [func(self, args)]
            for child in self._children:
                result += self._children[child].iterate(func, args, filter)
        return result