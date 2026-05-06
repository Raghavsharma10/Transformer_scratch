def createListRecursively(self,args):
        """
        This is an internal method to create the list of input files (or directories)
        recursively, starting at the provided directory or directories.
        """
        resultList = []
        dirDict = self.getDirectoryDictionary(args)
        for key in dirDict:
            for path,dirs,files in os.walk(key):    # Walk through the directory to find al	l input
                for d in dirs:
                    resultList.append(os.path.join(path,d))
                for f in files:    # Append the file if 'ALL' are allowed or the extension is allowed
                        pattern = dirDict[key].split(',')
                        if 'ALL' in pattern or os.path.splitext(f)[1] in pattern:
                            resultList.append(os.path.join(path,f))
        return list(set(resultList))