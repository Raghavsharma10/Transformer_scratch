def find_template(self, name=None, hv=None):
        """
        Return a list of templates that could have one or more elements.
        Args:
            name: name of the template to find.
            hv: the ID of the hypervisor to search the template in
        Returns:
            A list of templates object. If hv is None will return all the
            templates matching the name if every hypervisor type. Otherwise
            if name is None will return all templates of an hypervisor.
        Raises:
            ValidationError: if name and hv are None
        """
        if len(self.templates) <= 0:
            self.get_hypervisors()
        if name is not None and hv is not None:
            template_list = filter(
                lambda x: name in x.descr and x.hypervisor == self.hypervisors[hv], self.templates
            )
        elif name is not None and hv is None:
            template_list = filter(
                lambda x: name in x.descr, self.templates
            )
        elif name is None and hv is not None:
            template_list = filter(
                lambda x: x.hypervisor == self.hypervisors[hv], self.templates
            )
        else:
            raise Exception('Error, no pattern defined')
        if  sys.version_info.major < (3):
            return template_list
        else:
            return(list(template_list))