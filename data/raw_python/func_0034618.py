def add_elasticache_replication_group(self, replication_group, region):
        ''' Adds an ElastiCache replication group to the inventory and index '''

        # Only want available clusters unless all_elasticache_replication_groups is True
        if not self.all_elasticache_replication_groups and replication_group['Status'] != 'available':
            return

        # Select the best destination address (PrimaryEndpoint)
        dest = replication_group['NodeGroups'][0]['PrimaryEndpoint']['Address']

        if not dest:
            # Skip clusters we cannot address (e.g. private VPC subnet)
            return

        # Add to index
        self.index[dest] = [region, replication_group['ReplicationGroupId']]

        # Inventory: Group by ID (always a group of 1)
        if self.group_by_instance_id:
            self.inventory[replication_group['ReplicationGroupId']] = [dest]
            if self.nested_groups:
                self.push_group(self.inventory, 'instances', replication_group['ReplicationGroupId'])

        # Inventory: Group by region
        if self.group_by_region:
            self.push(self.inventory, region, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'regions', region)

        # Inventory: Group by availability zone (doesn't apply to replication groups)

        # Inventory: Group by node type (doesn't apply to replication groups)

        # Inventory: Group by VPC (information not available in the current
        # AWS API version for replication groups

        # Inventory: Group by security group (doesn't apply to replication groups)
        # Check this value in cluster level

        # Inventory: Group by engine (replication groups are always Redis)
        if self.group_by_elasticache_engine:
            self.push(self.inventory, 'elasticache_redis', dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'elasticache_engines', 'redis')

        # Global Tag: all ElastiCache clusters
        self.push(self.inventory, 'elasticache_replication_groups', replication_group['ReplicationGroupId'])

        host_info = self.get_host_info_dict_from_describe_dict(replication_group)

        self.inventory["_meta"]["hostvars"][dest] = host_info