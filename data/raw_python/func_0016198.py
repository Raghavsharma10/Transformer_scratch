def find(self, name):
        """
        Return a list of subset of VM that match the pattern name
        @param name (str): the vm name of the virtual machine
        @param name (Obj): the vm object that represent the virtual
                           machine (can be Pro or Smart)
        @return (list): the subset containing the serach result.
        """
        if name.__class__ is 'base.Server.Pro' or name.__class__ is 'base.Server.Smart':
            # print('DEBUG: matched VM object %s' % name.__class__)
            pattern = name.vm_name
        else:
            # print('DEBUG: matched Str Object %s' % name.__class__)
            pattern = name
        # 14/06/2013: since this method is called within a thread and I wont to pass the return objects with queue or
        # call back, I will allocate a list inside the Interface class object itself, which contain all of the vm found
        # 02/11/2015: this must be changed ASAP! it's a mess this way... what was I thinking??
        self.last_search_result = [vm for vm in self if pattern in vm.vm_name]
        return self.last_search_result