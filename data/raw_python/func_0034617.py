def add_elasticache_node(self, node, cluster, region):
        ''' Adds an ElastiCache node to the inventory and index, as long as
        it is addressable '''

        # Only want available nodes unless all_elasticache_nodes is True
        if not self.all_elasticache_nodes and node['CacheNodeStatus'] != 'available':
            return

        # Select the best destination address
        dest = node['Endpoint']['Address']

        if not dest:
            # Skip nodes we cannot address (e.g. private VPC subnet)
            return

        node_id = self.to_safe(cluster['CacheClusterId'] + '_' + node['CacheNodeId'])

        # Add to index
        self.index[dest] = [region, node_id]

        # Inventory: Group by node ID (always a group of 1)
        if self.group_by_instance_id:
            self.inventory[node_id] = [dest]
            if self.nested_groups:
                self.push_group(self.inventory, 'instances', node_id)

        # Inventory: Group by region
        if self.group_by_region:
            self.push(self.inventory, region, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'regions', region)

        # Inventory: Group by availability zone
        if self.group_by_availability_zone:
            self.push(self.inventory, cluster['PreferredAvailabilityZone'], dest)
            if self.nested_groups:
                if self.group_by_region:
                    self.push_group(self.inventory, region, cluster['PreferredAvailabilityZone'])
                self.push_group(self.inventory, 'zones', cluster['PreferredAvailabilityZone'])

        # Inventory: Group by node type
        if self.group_by_instance_type:
            type_name = self.to_safe('type_' + cluster['CacheNodeType'])
            self.push(self.inventory, type_name, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'types', type_name)

        # Inventory: Group by VPC (information not available in the current
        # AWS API version for ElastiCache)

        # Inventory: Group by security group
        if self.group_by_security_group:

            # Check for the existence of the 'SecurityGroups' key and also if
            # this key has some value. When the cluster is not placed in a SG
            # the query can return None here and cause an error.
            if 'SecurityGroups' in cluster and cluster['SecurityGroups'] is not None:
                for security_group in cluster['SecurityGroups']:
                    key = self.to_safe("security_group_" + security_group['SecurityGroupId'])
                    self.push(self.inventory, key, dest)
                    if self.nested_groups:
                        self.push_group(self.inventory, 'security_groups', key)

        # Inventory: Group by engine
        if self.group_by_elasticache_engine:
            self.push(self.inventory, self.to_safe("elasticache_" + cluster['Engine']), dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'elasticache_engines', self.to_safe("elasticache_" + cluster['Engine']))

        # Inventory: Group by parameter group (done at cluster level)

        # Inventory: Group by replication group (done at cluster level)

        # Inventory: Group by ElastiCache Cluster
        if self.group_by_elasticache_cluster:
            self.push(self.inventory, self.to_safe("elasticache_cluster_" + cluster['CacheClusterId']), dest)

        # Global Tag: all ElastiCache nodes
        self.push(self.inventory, 'elasticache_nodes', dest)

        host_info = self.get_host_info_dict_from_describe_dict(node)

        if dest in self.inventory["_meta"]["hostvars"]:
            self.inventory["_meta"]["hostvars"][dest].update(host_info)
        else:
            self.inventory["_meta"]["hostvars"][dest] = host_info