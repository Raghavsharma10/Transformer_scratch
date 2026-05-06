def walk(self,depth=0,fsNode=None) :
        """Note, this is a filtered walk"""
        if not fsNode :
            fsNode = FSNode(self.init_path,self.init_path,0)
            
        if fsNode.isdir() :
            if self.check_dir(fsNode) :
                if self.check_return(fsNode) :
                    yield fsNode                
                for n in fsNode.children() :
                    if n.islink() :
                        # currently we don't follow links
                        continue
                    for n2 in self.walk(depth+1,n) :
                        if self.check_return(n2) :
                            yield n2
        else :
            if self.check_file(fsNode) :
                if self.check_return(fsNode) :
                    yield fsNode
        raise StopIteration