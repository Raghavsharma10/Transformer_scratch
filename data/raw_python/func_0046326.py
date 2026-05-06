def pipe(cls, input):
        '''
        Creates a stdin source for Sh object chains.
        :param input: If string; open as filename, if iterable, send iterated content into next command,
        if none of the above assume file like object.

        To input a string as content source, wrap in iterable:

        print 'Lines in "my_input_string":', Sh.pipe([my_string_input]) | 'wc -l'
        :return: Stdin object for chaining Sh commands after.

        WARNING: If iterable is an endless generator command evaluation will never complete.
        '''
        if type(input) in (str, unicode):
            return cls.Stdin.from_file(input)
        try:
            return cls.Stdin.from_iterator(iter(input))
        except TypeError:
            pass
        return cls.Stdin.from_file(input)