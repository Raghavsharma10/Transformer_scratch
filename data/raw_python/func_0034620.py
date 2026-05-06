def get_instance_route53_names(self, instance):
        ''' Check if an instance is referenced in the records we have from
        Route53. If it is, return the list of domain names pointing to said
        instance. If nothing points to it, return an empty list. '''

        instance_attributes = [ 'public_dns_name', 'private_dns_name',
                                'ip_address', 'private_ip_address' ]

        name_list = set()

        for attrib in instance_attributes:
            try:
                value = getattr(instance, attrib)
            except AttributeError:
                continue

            if value in self.route53_records:
                name_list.update(self.route53_records[value])

        return list(name_list)