def load_modules(self):
        """Should instance interfaces and set them to interface, following `modules`"""
        if self.INTERFACES_MODULE is None:
            raise NotImplementedError("A module containing interfaces modules "
                                      "should be setup in INTERFACES_MODULE !")
        else:
            for module, permission in self.modules.items():
                i = getattr(self.INTERFACES_MODULE,
                            module).Interface(self, permission)
                self.interfaces[module] = i