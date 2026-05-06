def get_hypervisors(self):
        """
        Initialize the internal list containing each template available for each
        hypervisor.

        :return: [bool] True in case of success, otherwise False
        """
        json_scheme = self.gen_def_json_scheme('GetHypervisors')
        json_obj = self.call_method_post(method='GetHypervisors', json_scheme=json_scheme)
        self.json_templates = json_obj
        d = dict(json_obj)
        for elem in d['Value']:
            hv = self.hypervisors[elem['HypervisorType']]
            for inner_elem in elem['Templates']:
                o = Template(hv)
                o.template_id = inner_elem['Id']
                o.descr = inner_elem['Description']
                o.id_code = inner_elem['IdentificationCode']
                o.name = inner_elem['Name']
                o.enabled = inner_elem['Enabled']
                if hv != 'SMART':
                    for rb in inner_elem['ResourceBounds']:
                        resource_type = rb['ResourceType']
                        if resource_type == 1:
                            o.resource_bounds.max_cpu = rb['Max']
                        if resource_type == 2:
                            o.resource_bounds.max_memory = rb['Max']
                        if resource_type == 3:
                            o.resource_bounds.hdd0 = rb['Max']
                        if resource_type == 7:
                            o.resource_bounds.hdd1 = rb['Max']
                        if resource_type == 8:
                            o.resource_bounds.hdd2 = rb['Max']
                        if resource_type == 9:
                            o.resource_bounds.hdd3 = rb['Max']
                self.templates.append(o)
        return True if json_obj['Success'] is 'True' else False