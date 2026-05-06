def getExtensionList(self,extensions):
        """
        This is an internal method that transforms the comma-separated extensions string
        into a list of extensions, e.g., "ext1,ext2,ext3" gets turned into ['.ext1','.ext2','.ext3'].
        If MapConstants.placeholderNoExtensionFilter is part of the string, the resulting list
        will also contain '', i.e., files without extensions are permitted.
        """
        basicList = extensions.split(',')
        extensionList = []
        for ext in basicList:
            if ext == MapConstants.placeholderNoExtensionFilter:
                # Files without an extension are permitted:
                extensionList.append('')
            elif ext != '':
                # The '.' is prepended if ext does not start with '.' already:
                extWithDot = ext if ext.startswith('.') else '.'+ext
                extensionList.append(extWithDot)
        return list(set(extensionList))