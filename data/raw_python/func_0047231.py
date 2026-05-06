def generate_package_content(self, package):
        """Generate package.rst text content.

        ::

            {{ package_name }}
            ==================

            .. automodule:: {{ package_name }}
                :members:

            sub packages and modules
            ------------------------

            .. toctree::
               :maxdepth: 1

                {{ sub_package_name1 }} <{{ sub_package_name1 }}/__init__>
                {{ sub_package_name2 }} <{{ sub_package_name2 }}/__init__>
                {{ sub_module_name1}} <{{ sub_module_name1}}>
                {{ sub_module_name2}} <{{ sub_module_name2}}>

        """
        if isinstance(package, Package):
            return package.render(ignored_package=self.ignored_package)
        else:  # pragma: no cover
            raise Exception("%r is not a Package object" % package)