def get_imports(self, module, return_fqn=False):
        """return set of imported modules that are in self
        :param module: PyModule
        :return: (set - str) of path names
        """
        # print('####', module.fqn)
        # print(self.by_name.keys(), '\n\n')
        imports = set()
        raw_imports = ast_imports(module.path)
        for import_entry in raw_imports:
            # join 'from' and 'import' part of import statement
            full = ".".join(s for s in import_entry[:2] if s)

            import_level = import_entry[3]
            if import_level:
                # intra package imports
                intra = '.'.join(module.fqn[:-import_level] + [full])
                imported = self._get_imported_module(intra)
            else:
                imported = self._get_imported_module(full)

            if imported:
                if return_fqn:
                    imports.add('.'.join(imported.fqn))
                else:
                    imports.add(imported.path)
        return imports