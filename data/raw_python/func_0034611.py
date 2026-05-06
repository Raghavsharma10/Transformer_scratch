def get_elasticache_replication_groups_by_region(self, region):
        ''' Makes an AWS API call to the list of ElastiCache replication groups
        in a particular region.'''

        # ElastiCache boto module doesn't provide a get_all_intances method,
        # that's why we need to call describe directly (it would be called by
        # the shorthand method anyway...)
        try:
            conn = elasticache.connect_to_region(region)
            if conn:
                response = conn.describe_replication_groups()

        except boto.exception.BotoServerError as e:
            error = e.reason

            if e.error_code == 'AuthFailure':
                error = self.get_auth_error_message()
            if not e.reason == "Forbidden":
                error = "Looks like AWS ElastiCache [Replication Groups] is down:\n%s" % e.message
            self.fail_with_error(error, 'getting ElastiCache clusters')

        try:
            # Boto also doesn't provide wrapper classes to ReplicationGroups
            # Because of that wo can't make use of the get_list method in the
            # AWSQueryConnection. Let's do the work manually
            replication_groups = response['DescribeReplicationGroupsResponse']['DescribeReplicationGroupsResult']['ReplicationGroups']

        except KeyError as e:
            error = "ElastiCache [Replication Groups] query to AWS failed (unexpected format)."
            self.fail_with_error(error, 'getting ElastiCache clusters')

        for replication_group in replication_groups:
            self.add_elasticache_replication_group(replication_group, region)