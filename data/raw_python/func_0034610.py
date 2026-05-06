def get_elasticache_clusters_by_region(self, region):
        ''' Makes an AWS API call to the list of ElastiCache clusters (with
        nodes' info) in a particular region.'''

        # ElastiCache boto module doesn't provide a get_all_intances method,
        # that's why we need to call describe directly (it would be called by
        # the shorthand method anyway...)
        try:
            conn = elasticache.connect_to_region(region)
            if conn:
                # show_cache_node_info = True
                # because we also want nodes' information
                response = conn.describe_cache_clusters(None, None, None, True)

        except boto.exception.BotoServerError as e:
            error = e.reason

            if e.error_code == 'AuthFailure':
                error = self.get_auth_error_message()
            if not e.reason == "Forbidden":
                error = "Looks like AWS ElastiCache is down:\n%s" % e.message
            self.fail_with_error(error, 'getting ElastiCache clusters')

        try:
            # Boto also doesn't provide wrapper classes to CacheClusters or
            # CacheNodes. Because of that wo can't make use of the get_list
            # method in the AWSQueryConnection. Let's do the work manually
            clusters = response['DescribeCacheClustersResponse']['DescribeCacheClustersResult']['CacheClusters']

        except KeyError as e:
            error = "ElastiCache query to AWS failed (unexpected format)."
            self.fail_with_error(error, 'getting ElastiCache clusters')

        for cluster in clusters:
            self.add_elasticache_cluster(cluster, region)