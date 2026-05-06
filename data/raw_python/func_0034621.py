def get_host_info_dict_from_describe_dict(self, describe_dict):
        ''' Parses the dictionary returned by the API call into a flat list
            of parameters. This method should be used only when 'describe' is
            used directly because Boto doesn't provide specific classes. '''

        # I really don't agree with prefixing everything with 'ec2'
        # because EC2, RDS and ElastiCache are different services.
        # I'm just following the pattern used until now to not break any
        # compatibility.

        host_info = {}
        for key in describe_dict:
            value = describe_dict[key]
            key = self.to_safe('ec2_' + self.uncammelize(key))

            # Handle complex types

            # Target: Memcached Cache Clusters
            if key == 'ec2_configuration_endpoint' and value:
                host_info['ec2_configuration_endpoint_address'] = value['Address']
                host_info['ec2_configuration_endpoint_port'] = value['Port']

            # Target: Cache Nodes and Redis Cache Clusters (single node)
            if key == 'ec2_endpoint' and value:
                host_info['ec2_endpoint_address'] = value['Address']
                host_info['ec2_endpoint_port'] = value['Port']

            # Target: Redis Replication Groups
            if key == 'ec2_node_groups' and value:
                host_info['ec2_endpoint_address'] = value[0]['PrimaryEndpoint']['Address']
                host_info['ec2_endpoint_port'] = value[0]['PrimaryEndpoint']['Port']
                replica_count = 0
                for node in value[0]['NodeGroupMembers']:
                    if node['CurrentRole'] == 'primary':
                        host_info['ec2_primary_cluster_address'] = node['ReadEndpoint']['Address']
                        host_info['ec2_primary_cluster_port'] = node['ReadEndpoint']['Port']
                        host_info['ec2_primary_cluster_id'] = node['CacheClusterId']
                    elif node['CurrentRole'] == 'replica':
                        host_info['ec2_replica_cluster_address_'+ str(replica_count)] = node['ReadEndpoint']['Address']
                        host_info['ec2_replica_cluster_port_'+ str(replica_count)] = node['ReadEndpoint']['Port']
                        host_info['ec2_replica_cluster_id_'+ str(replica_count)] = node['CacheClusterId']
                        replica_count += 1

            # Target: Redis Replication Groups
            if key == 'ec2_member_clusters' and value:
                host_info['ec2_member_clusters'] = ','.join([str(i) for i in value])

            # Target: All Cache Clusters
            elif key == 'ec2_cache_parameter_group':
                host_info["ec2_cache_node_ids_to_reboot"] = ','.join([str(i) for i in value['CacheNodeIdsToReboot']])
                host_info['ec2_cache_parameter_group_name'] = value['CacheParameterGroupName']
                host_info['ec2_cache_parameter_apply_status'] = value['ParameterApplyStatus']

            # Target: Almost everything
            elif key == 'ec2_security_groups':

                # Skip if SecurityGroups is None
                # (it is possible to have the key defined but no value in it).
                if value is not None:
                    sg_ids = []
                    for sg in value:
                        sg_ids.append(sg['SecurityGroupId'])
                    host_info["ec2_security_group_ids"] = ','.join([str(i) for i in sg_ids])

            # Target: Everything
            # Preserve booleans and integers
            elif type(value) in [int, bool]:
                host_info[key] = value

            # Target: Everything
            # Sanitize string values
            elif isinstance(value, six.string_types):
                host_info[key] = value.strip()

            # Target: Everything
            # Replace None by an empty string
            elif type(value) == type(None):
                host_info[key] = ''

            else:
                # Remove non-processed complex types
                pass

        return host_info