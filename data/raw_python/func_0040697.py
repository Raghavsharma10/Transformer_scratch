def DirContains(self,f) :
        """ Matches dirs that have a child that matches filter f"""
        def match(fsNode) :
            if not fsNode.isdir() : return False 
            for c in fsNode.children() :
                if f(c) : return True
            return False
        return self.make_return(match)