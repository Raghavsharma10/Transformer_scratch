def getDirectoryDictionary(self,args):
        """
        This is an internal method to compute a dictionary containing
        all the directories that potentially contain (more) input.
        The dictionary 'table' indicates which of its contents are part of the input:

        table[directory] == 'ALL' means that its entire content must be mapped
        table[directory] == 'ext' means that only content with the extension 'ext' are mapped
        table[directory] == 'ext1,ext2' means that only content with either extension 'ext1'
                            or 'ext2' are mapped

        Note that there is a special symbol MapConstants.placeholderNoExtensionFilter
        that enables the filtering for files without an extension.
        """
        table = {}
        for element in args.path:
            # If the element is a directory, the dictionary entry is set to 'ALL':
            if os.path.isdir(element):
                if element not in table:
                    table[element] = 'ALL'
            # If the element is a file, its extension is added to the dictionary entry:
            else:
                path = os.path.dirname(element)
                extension = os.path.splitext(element)[1]
                # If there is no extension, a '.' was missing. This happens when calling:
                # map -r [command] path/to/folder/*ext
                # and the folder 'folder' DOES NOT contain files with the extension 'ext'.
                # In this case, there is no wildcard expansion and we must create the
                # proper extension manually:
                if extension == '':
                    extension = '.' + os.path.basename(element).split("*")[-1]
                if path not in table:
                    table[path] = extension
                elif table[path] != 'ALL' and extension not in table[path]:
                    table[path] = table[path] + "," + extension
        return table