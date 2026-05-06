def read_from_file(filename):
        """
           Arguments:
            | ``filename``  --  the filename of the input file

           Use as follows::

             >>> if = CP2KInputFile.read_from_file("somefile.inp")
             >>> for section in if:
             ...     print section.name
        """
        with open(filename) as f:
            result = CP2KInputFile()
            try:
                while True:
                    result.load_children(f)
            except EOFError:
                pass
        return result