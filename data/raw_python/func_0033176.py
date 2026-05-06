def getTmpFilename(self, tmp_dir=None, prefix='tmp', suffix='.txt',
                       include_class_id=False, result_constructor=FilePath):
        """ Return a temp filename

            tmp_dir: directory where temporary files will be stored
            prefix: text to append to start of file name
            suffix: text to append to end of file name
            include_class_id: if True, will append a class identifier (built
             from the class name) to the filename following prefix. This is
             False by default b/c there is some string processing overhead
             in getting the class name. This will probably be most useful for
             testing: if temp files are being left behind by tests, you can
             turn this on in here (temporarily) to find out which tests are
             leaving the temp files.
            result_constructor: the constructor used to build the result
             (default: cogent.app.parameters.FilePath). Note that joining
             FilePath objects with one another or with strings, you must use
             the + operator. If this causes trouble, you can pass str as the
             the result_constructor.
        """

        # check not none
        if not tmp_dir:
            tmp_dir = self.TmpDir
        # if not current directory, append "/" if not already on path
        elif not tmp_dir.endswith("/"):
            tmp_dir += "/"

        if include_class_id:
            # Append the classname to the prefix from the class name
            # so any problematic temp files can be associated with
            # the class that created them. This should be especially
            # useful for testing, but is turned off by default to
            # avoid the string-parsing overhead.
            class_id = str(self.__class__())
            prefix = ''.join([prefix,
                              class_id[class_id.rindex('.') + 1:
                                       class_id.index(' ')]])

        try:
            mkdir(tmp_dir)
        except OSError:
            # Directory already exists
            pass
        # note: it is OK to join FilePath objects with +
        return result_constructor(tmp_dir) + result_constructor(prefix) + \
            result_constructor(''.join([choice(_all_chars)
                                        for i in range(self.TmpNameLen)])) +\
            result_constructor(suffix)