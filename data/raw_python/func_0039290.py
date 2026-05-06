def removeBinder(self, name):
        """Remove a binder from a table
        """
        root = self.etree
        
        t_bindings = root.find('bindings')
        
        t_binder = t_bindings.find(name)

        if t_binder :
            t_bindings.remove(t_binder)
            return True
        
        return False