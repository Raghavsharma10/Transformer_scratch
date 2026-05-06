def add(self, function, kind='add', at_pos='end',*args, **kwargs):
        """Add a function to custom processing queue. 
        
        Custom functions are applied automatically to associated
        pysat instrument whenever instrument.load command called.

        Parameters
        ----------
            function : string or function object
                name of function or function object to be added to queue
            
            kind : {'add', 'modify', 'pass}
                add 
                    Adds data returned from function to instrument object. 
                    A copy of pysat instrument object supplied to routine. 
                modify  
                    pysat instrument object supplied to routine. Any and all
                    changes to object are retained.
                pass  
                    A copy of pysat object is passed to function. No 
                    data is accepted from return.
                       
            at_pos : string or int 
                insert at position. (default, insert at end).
            args : extra arguments
                extra arguments are passed to the custom function (once)
            kwargs : extra keyword arguments
                extra keyword args are passed to the custom function (once)

        Note
        ----
        Allowed `add` function returns:

        - {'data' : pandas Series/DataFrame/array_like,
          'units' : string/array_like of strings, 
          'long_name' : string/array_like of strings,
          'name' : string/array_like of strings (iff data array_like)} 

        - pandas DataFrame, names of columns are used

        - pandas Series, .name required 

        - (string/list of strings, numpy array/list of arrays) 
        """

        if isinstance(function, str):
            # convert string to function object
            function=eval(function)

        if (at_pos == 'end') | (at_pos == len(self._functions)):
            # store function object
            self._functions.append(function)
            self._args.append(args)
            self._kwargs.append(kwargs)
            self._kind.append(kind.lower())
        elif at_pos < len(self._functions):
            # user picked a specific location to insert
            self._functions.insert(at_pos, function)
            self._args.insert(at_pos, args)
            self._kwargs.insert(at_pos, kwargs)
            self._kind.insert(at_pos, kind)
        else:
            raise TypeError('Must enter an index between 0 and %i' %
                            len(self._functions))