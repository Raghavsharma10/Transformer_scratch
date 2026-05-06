def shadow(self,new_root,visitor) :
        """ Runs through the query, creating a clone directory structure in the new_root. Then applies process"""
        for n in self.walk() :
            sn = n.clone(new_root)
            if n.isdir() :
                visitor.process_dir(n,sn)
            else :
                visitor.process_file(n,sn)