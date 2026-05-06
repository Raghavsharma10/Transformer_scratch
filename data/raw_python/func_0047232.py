def generate_module_content(self, module):
        """Generate module.rst text content.

        ::

            {{ module_name }}
            =================

            .. automodule:: {{ module_fullname }}
                :members:
        """
        if isinstance(module, Module):
            return module.render()
        else:  # pragma: no cover
            raise Exception("%r is not a Module object" % module)