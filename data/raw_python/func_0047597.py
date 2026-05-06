def walk(self):
        """
        A generator that walking through all sub packages and sub modules.

        1. current package object (包对象)
        2. current package's parent (当前包对象的母包)
        3. list of sub packages (所有子包)
        4. list of sub modules (所有模块)
        """
        yield (
            self,
            self.parent,
            list(self.sub_packages.values()),
            list(self.sub_modules.values()),
        )

        for pkg in self.sub_packages.values():
            for things in pkg.walk():
                yield things