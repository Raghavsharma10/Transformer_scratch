def _get_imported_module(self, module_name):
        """try to get imported module reference by its name"""
        # if imported module on module_set add to list
        imp_mod = self.by_name.get(module_name)
        if imp_mod:
            return imp_mod

        # last part of import section might not be a module
        # remove last section
        no_obj = module_name.rsplit('.', 1)[0]
        imp_mod2 = self.by_name.get(no_obj)
        if imp_mod2:
            return imp_mod2

        # special case for __init__
        if module_name in self.pkgs:
            pkg_name = module_name  + ".__init__"
            return self.by_name[pkg_name]

        if no_obj in self.pkgs:
            pkg_name = no_obj +  ".__init__"
            return self.by_name[pkg_name]