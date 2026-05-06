def create_install_template_skin(self):
        """
        Create an example ckan extension for this environment and install it
        """
        ckan_extension_template(self.name, self.target)
        self.install_package_develop('ckanext-' + self.name + 'theme')