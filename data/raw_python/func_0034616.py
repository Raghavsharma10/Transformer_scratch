def add_elasticache_cluster(self, cluster, region):
        ''' Adds an ElastiCache cluster to the inventory and index, as long as
        it's nodes are addressable '''

        # Only want available clusters unless all_elasticache_clusters is True
        if not self.all_elasticache_clusters and cluster['CacheClusterStatus'] != 'available':
            return

        # Select the best destination address
        if 'ConfigurationEndpoint' in cluster and cluster['ConfigurationEndpoint']:
            # Memcached cluster
            dest = cluster['ConfigurationEndpoint']['Address']
            is_redis = False
        else:
            # Redis sigle node cluster
            # Because all Redis clusters are single nodes, we'll merge the
            # info from the cluster with info about the node
            dest = cluster['CacheNodes'][0]['Endpoint']['Address']
            is_redis = True

        if not dest:
            # Skip clusters we cannot address (e.g. private VPC subnet)
            return

        # Add to index
        self.index[dest] = [region, cluster['CacheClusterId']]

        # Inventory: Group by instance ID (always a group of 1)
        if self.group_by_instance_id:
            self.inventory[cluster['CacheClusterId']] = [dest]
            if self.nested_groups:
                self.push_group(self.inventory, 'instances', cluster['CacheClusterId'])

        # Inventory: Group by region
        if self.group_by_region and not is_redis:
            self.push(self.inventory, region, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'regions', region)

        # Inventory: Group by availability zone
        if self.group_by_availability_zone and not is_redis:
            self.push(self.inventory, cluster['PreferredAvailabilityZone'], dest)
            if self.nested_groups:
                if self.group_by_region:
                    self.push_group(self.inventory, region, cluster['PreferredAvailabilityZone'])
                self.push_group(self.inventory, 'zones', cluster['PreferredAvailabilityZone'])

        # Inventory: Group by node type
        if self.group_by_instance_type and not is_redis:
            type_name = self.to_safe('type_' + cluster['CacheNodeType'])
            self.push(self.inventory, type_name, dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'types', type_name)

        # Inventory: Group by VPC (information not available in the current
        # AWS API version for ElastiCache)

        # Inventory: Group by security group
        if self.group_by_security_group and not is_redis:

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
        if self.group_by_elasticache_engine and not is_redis:
            self.push(self.inventory, self.to_safe("elasticache_" + cluster['Engine']), dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'elasticache_engines', self.to_safe(cluster['Engine']))

        # Inventory: Group by parameter group
        if self.group_by_elasticache_parameter_group:
            self.push(self.inventory, self.to_safe("elasticache_parameter_group_" + cluster['CacheParameterGroup']['CacheParameterGroupName']), dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'elasticache_parameter_groups', self.to_safe(cluster['CacheParameterGroup']['CacheParameterGroupName']))

        # Inventory: Group by replication group
        if self.group_by_elasticache_replication_group and 'ReplicationGroupId' in cluster and cluster['ReplicationGroupId']:
            self.push(self.inventory, self.to_safe("elasticache_replication_group_" + cluster['ReplicationGroupId']), dest)
            if self.nested_groups:
                self.push_group(self.inventory, 'elasticache_replication_groups', self.to_safe(cluster['ReplicationGroupId']))

        # Global Tag: all ElastiCache clusters
        self.push(self.inventory, 'elasticache_clusters', cluster['CacheClusterId'])

        host_info = self.get_host_info_dict_from_describe_dict(cluster)

        self.inventory["_meta"]["hostvars"][dest] = host_info

        # Add the nodes
        for node in cluster['CacheNodes']:
            self.add_elasticache_node(node, cluster, region)