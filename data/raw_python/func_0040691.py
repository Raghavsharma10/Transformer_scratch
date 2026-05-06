def children(self) :
        "If the FSNode is a directory, returns a list of the children"
        if not self.isdir() : raise Exception("FSQuery tried to return the children of a node which is not a directory : %s" % self.abs)
        return [FSNode(self.abs + "/" + x,self.root,self.depth+1) for x in os.listdir(self.abs)]