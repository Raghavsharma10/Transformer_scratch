def createList(self,args):
        """
        This is an internal method to create the list of input files (or directories)
        contained in the provided directory or directories.
        """
        resultList = []
        if len(args.path) == 1 and os.path.isdir(args.path[0]):
            resultList = [os.path.join(args.path[0], f) for f in os.listdir(args.path[0])]
        else:    # If there are multiple items, wildcard expansion has already created the list of files
            resultList = args.path
        return list(set(resultList))