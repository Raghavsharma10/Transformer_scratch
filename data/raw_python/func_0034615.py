def add_rds_instance(self, instance, region):
        ''' Adds an RDS instance to the inventory and index, as long as it is
        addressable '''

        # Only want available instances unless all_rds_instances is True
        if not self.all_rds_instances and instance.status != 'available':
            return

        # Select the best destination address
        dest = instance.endpoint[0]

        if not dest:
            # Skip instances we cannot address (e.g. private VPC subnet)
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
            self.push(self.inventory, instance.availability_zone, dest)
            if self.nested_groups:
                if self.group_by_region:
                    self.push_group(self.inventory, region, instance.availability_zone)
                self.push_group(self.inventory, 'zones', instance.availability_zone)

        # Inventory: Group by instance type
        if self.group_by_instance_type:
            type_name = self.to_safe('type_' + instance.instance_class)
            self.push(self.inventory, type_name, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'types', type_name)

        # Inventory: Group by VPC
        if self.group_by_vpc_id and instance.subnet_group and instance.subnet_group.vpc_id:
            vpc_id_name = self.to_safe('vpc_id_' + instance.subnet_group.vpc_id)
            self.push(self.inventory, vpc_id_name, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'vpcs', vpc_id_name)

        # Inventory: Group by security group
        if self.group_by_security_group:
            try:
                if instance.security_group:
                    key = self.to_safe("security_group_" + instance.security_group.name)
                    self.push(self.inventory, key, dest)
                    if self.nested_groups:
                        self.push_group(self.inventory, 'security_groups', key)

            except AttributeError:
                self.fail_with_error('\n'.join(['Package boto seems a bit older.', 
                                            'Please upgrade boto >= 2.3.0.']))


        # Inventory: Group by engine
        if self.group_by_rds_engine:
            self.push(self.inventory, self.to_safe("rds_" + instance.engine), dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'rds_engines', self.to_safe("rds_" + instance.engine))

        # Inventory: Group by parameter group
        if self.group_by_rds_parameter_group:
            self.push(self.inventory, self.to_safe("rds_parameter_group_" + instance.parameter_group.name), dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'rds_parameter_groups', self.to_safe("rds_parameter_group_" + instance.parameter_group.name))

        # Global Tag: all RDS instances
        self.push(self.inventory, 'rds', dest)

        self.inventory["_meta"]["hostvars"][dest] = self.get_host_info_dict_from_instance(instance)