def data_mod(self, *args, **kwargs):
        """
        Register a function to modify data of member Instruments.

        The function is not partially applied to modify member data.

        When the Constellation receives a function call to register a function for data modification,
        it passes the call to each instrument and registers it in the instrument's pysat.Custom queue.

        (Wraps pysat.Custom.add; documentation of that function is
        reproduced here.)

        Parameters
        ----------
            function : string or function object
                name of function or function object to be added to queue

            kind : {'add, 'modify', 'pass'}
                add
                    Adds data returned from fuction to instrument object.
                modify
                    pysat instrument object supplied to routine. Any and all
                    changes to object are retained.
                pass
                    A copy of pysat object is passed to function. No
                    data is accepted from return.

            at_pos : string or int
                insert at position. (default, insert at end).
            args : extra arguments

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

        for instrument in self.instruments:
            instrument.custom.add(*args, **kwargs)