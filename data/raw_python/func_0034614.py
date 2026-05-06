def add_instance(self, instance, region):
        ''' Adds an instance to the inventory and index, as long as it is
        addressable '''

        # Only return instances with desired instance states
        if instance.state not in self.ec2_instance_states:
            return

        # Select the best destination address
        if instance.subnet_id:
            dest = getattr(instance, self.vpc_destination_variable, None)
            if dest is None:
                dest = getattr(instance, 'tags').get(self.vpc_destination_variable, None)
        else:
            dest = getattr(instance, self.destination_variable, None)
            if dest is None:
                dest = getattr(instance, 'tags').get(self.destination_variable, None)

        if not dest:
            # Skip instances we cannot address (e.g. private VPC subnet)
            return

        # if we only want to include hosts that match a pattern, skip those that don't
        if self.pattern_include and not self.pattern_include.match(dest):
            return

        # if we need to exclude hosts that match a pattern, skip those
        if self.pattern_exclude and self.pattern_exclude.match(dest):
            return

        # Add to index
        self.index[dest] = [region, instance.id]

        # Inventory: Group by instance ID (always a group of 1)
        if self.group_by_instance_id:
            self.inventory[instance.id] = [dest]
            if self.nested_groups:
                self.push_group(self.inventory, 'instances', instance.id)

        # Inventory: Group by region
        if self.group_by_region:
            self.push(self.inventory, region, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'regions', region)

        # Inventory: Group by availability zone
        if self.group_by_availability_zone:
            self.push(self.inventory, instance.placement, dest)
            if self.nested_groups:
                if self.group_by_region:
                    self.push_group(self.inventory, region, instance.placement)
                self.push_group(self.inventory, 'zones', instance.placement)

        # Inventory: Group by Amazon Machine Image (AMI) ID
        if self.group_by_ami_id:
            ami_id = self.to_safe(instance.image_id)
            self.push(self.inventory, ami_id, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'images', ami_id)

        # Inventory: Group by instance type
        if self.group_by_instance_type:
            type_name = self.to_safe('type_' + instance.instance_type)
            self.push(self.inventory, type_name, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'types', type_name)

        # Inventory: Group by key pair
        if self.group_by_key_pair and instance.key_name:
            key_name = self.to_safe('key_' + instance.key_name)
            self.push(self.inventory, key_name, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'keys', key_name)

        # Inventory: Group by VPC
        if self.group_by_vpc_id and instance.vpc_id:
            vpc_id_name = self.to_safe('vpc_id_' + instance.vpc_id)
            self.push(self.inventory, vpc_id_name, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'vpcs', vpc_id_name)

        # Inventory: Group by security group
        if self.group_by_security_group:
            try:
                for group in instance.groups:
                    key = self.to_safe("security_group_" + group.name)
                    self.push(self.inventory, key, dest)
                    if self.nested_groups:
                        self.push_group(self.inventory, 'security_groups', key)
            except AttributeError:
                self.fail_with_error('\n'.join(['Package boto seems a bit older.', 
                                            'Please upgrade boto >= 2.3.0.']))

        # Inventory: Group by tag keys
        if self.group_by_tag_keys:
            for k, v in instance.tags.items():
                if self.expand_csv_tags and v and ',' in v:
                    values = map(lambda x: x.strip(), v.split(','))
                else:
                    values = [v]

                for v in values:
                    if v:
                        key = self.to_safe("tag_" + k + "=" + v)
                    else:
                        key = self.to_safe("tag_" + k)
                    self.push(self.inventory, key, dest)
                    if self.nested_groups:
                        self.push_group(self.inventory, 'tags', self.to_safe("tag_" + k))
                        if v:
                            self.push_group(self.inventory, self.to_safe("tag_" + k), key)

        # Inventory: Group by Route53 domain names if enabled
        if self.route53_enabled and self.group_by_route53_names:
            route53_names = self.get_instance_route53_names(instance)
            for name in route53_names:
                self.push(self.inventory, name, dest)
                if self.nested_groups:
                    self.push_group(self.inventory, 'route53', name)

        # Global Tag: instances without tags
        if self.group_by_tag_none and len(instance.tags) == 0:
            self.push(self.inventory, 'tag_none', dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'tags', 'tag_none')

        # Global Tag: tag all EC2 instances
        self.push(self.inventory, 'ec2', dest)

        self.inventory["_meta"]["hostvars"][dest] = self.get_host_info_dict_from_instance(instance)