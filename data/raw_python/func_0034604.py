def read_settings(self, configfile):
        ''' Reads the settings from the ec2.ini file '''
        if six.PY3:
            config = configparser.ConfigParser()
        else:
            config = configparser.SafeConfigParser()
        config.read(configfile)

        # is eucalyptus?
        self.eucalyptus_host = None
        self.eucalyptus = False
        if config.has_option('ec2', 'eucalyptus'):
            self.eucalyptus = config.getboolean('ec2', 'eucalyptus')
        if self.eucalyptus and config.has_option('ec2', 'eucalyptus_host'):
            self.eucalyptus_host = config.get('ec2', 'eucalyptus_host')

        # Regions
        self.regions = []
        configRegions = config.get('ec2', 'regions')
        configRegions_exclude = config.get('ec2', 'regions_exclude')
        if (configRegions == 'all'):
            if self.eucalyptus_host:
                self.regions.append(boto.connect_euca(host=self.eucalyptus_host).region.name)
            else:
                for regionInfo in ec2.regions():
                    if regionInfo.name not in configRegions_exclude:
                        self.regions.append(regionInfo.name)
        else:
            self.regions = configRegions.split(",")

        # Destination addresses
        self.destination_variable = config.get('ec2', 'destination_variable')
        self.vpc_destination_variable = config.get('ec2',
                                                   'vpc_destination_variable')

        # Route53
        self.route53_enabled = config.getboolean('ec2', 'route53')
        self.route53_excluded_zones = []
        if config.has_option('ec2', 'route53_excluded_zones'):
            self.route53_excluded_zones.extend(
                config.get('ec2', 'route53_excluded_zones', '').split(','))

        # Include RDS instances?
        self.rds_enabled = True
        if config.has_option('ec2', 'rds'):
            self.rds_enabled = config.getboolean('ec2', 'rds')

        # Include ElastiCache instances?
        self.elasticache_enabled = True
        if config.has_option('ec2', 'elasticache'):
            self.elasticache_enabled = config.getboolean('ec2', 'elasticache')

        # Return all EC2 instances?
        if config.has_option('ec2', 'all_instances'):
            self.all_instances = config.getboolean('ec2', 'all_instances')
        else:
            self.all_instances = False

        # Instance states to be gathered in inventory. Default is 'running'.
        # Setting 'all_instances' to 'yes' overrides this option.
        ec2_valid_instance_states = [
            'pending',
            'running',
            'shutting-down',
            'terminated',
            'stopping',
            'stopped'
        ]
        self.ec2_instance_states = []
        if self.all_instances:
            self.ec2_instance_states = ec2_valid_instance_states
        elif config.has_option('ec2', 'instance_states'):
            for instance_state in config.get('ec2',
                                             'instance_states').split(','):
                instance_state = instance_state.strip()
                if instance_state not in ec2_valid_instance_states:
                    continue
                self.ec2_instance_states.append(instance_state)
        else:
            self.ec2_instance_states = ['running']

        # Return all RDS instances? (if RDS is enabled)
        if config.has_option('ec2', 'all_rds_instances') and self.rds_enabled:
            self.all_rds_instances = config.getboolean('ec2',
                                                       'all_rds_instances')
        else:
            self.all_rds_instances = False

        # Return all ElastiCache replication groups?
        # (if ElastiCache is enabled)
        if (config.has_option('ec2', 'all_elasticache_replication_groups')
                and self.elasticache_enabled):
            self.all_elasticache_replication_groups = config.getboolean(
                'ec2', 'all_elasticache_replication_groups')
        else:
            self.all_elasticache_replication_groups = False

        # Return all ElastiCache clusters? (if ElastiCache is enabled)
        if (config.has_option('ec2', 'all_elasticache_clusters')
                and self.elasticache_enabled):
            self.all_elasticache_clusters = config.getboolean(
                'ec2', 'all_elasticache_clusters')
        else:
            self.all_elasticache_clusters = False

        # Return all ElastiCache nodes? (if ElastiCache is enabled)
        if config.has_option('ec2', 'all_elasticache_nodes') and self.elasticache_enabled:
            self.all_elasticache_nodes = config.getboolean('ec2', 'all_elasticache_nodes')
        else:
            self.all_elasticache_nodes = False

        # boto configuration profile (prefer CLI argument)
        if config.has_option('ec2', 'boto_profile') and not self.boto_profile:
            self.boto_profile = config.get('ec2', 'boto_profile')

        # Cache related
        cache_dir = os.path.expanduser(config.get('ec2', 'cache_path'))
        if self.boto_profile:
            cache_dir = os.path.join(cache_dir, 'profile_' + self.boto_profile)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.cache_path_cache = cache_dir + "/ansible-ec2.cache"
        self.cache_path_index = cache_dir + "/ansible-ec2.index"
        self.cache_max_age = config.getint('ec2', 'cache_max_age')

        if config.has_option('ec2', 'expand_csv_tags'):
            self.expand_csv_tags = config.getboolean('ec2', 'expand_csv_tags')
        else:
            self.expand_csv_tags = False

        # Configure nested groups instead of flat namespace.
        if config.has_option('ec2', 'nested_groups'):
            self.nested_groups = config.getboolean('ec2', 'nested_groups')
        else:
            self.nested_groups = False

        # Replace dash or not in group names
        if config.has_option('ec2', 'replace_dash_in_groups'):
            self.replace_dash_in_groups = config.getboolean('ec2', 'replace_dash_in_groups')
        else:
            self.replace_dash_in_groups = True

        # Configure which groups should be created.
        group_by_options = [
            'group_by_instance_id',
            'group_by_region',
            'group_by_availability_zone',
            'group_by_ami_id',
            'group_by_instance_type',
            'group_by_key_pair',
            'group_by_vpc_id',
            'group_by_security_group',
            'group_by_tag_keys',
            'group_by_tag_none',
            'group_by_route53_names',
            'group_by_rds_engine',
            'group_by_rds_parameter_group',
            'group_by_elasticache_engine',
            'group_by_elasticache_cluster',
            'group_by_elasticache_parameter_group',
            'group_by_elasticache_replication_group',
        ]
        for option in group_by_options:
            if config.has_option('ec2', option):
                setattr(self, option, config.getboolean('ec2', option))
            else:
                setattr(self, option, True)

        # Do we need to just include hosts that match a pattern?
        try:
            pattern_include = config.get('ec2', 'pattern_include')
            if pattern_include and len(pattern_include) > 0:
                self.pattern_include = re.compile(pattern_include)
            else:
                self.pattern_include = None
        except configparser.NoOptionError:
            self.pattern_include = None

        # Do we need to exclude hosts that match a pattern?
        try:
            pattern_exclude = config.get('ec2', 'pattern_exclude');
            if pattern_exclude and len(pattern_exclude) > 0:
                self.pattern_exclude = re.compile(pattern_exclude)
            else:
                self.pattern_exclude = None
        except configparser.NoOptionError:
            self.pattern_exclude = None

        # Instance filters (see boto and EC2 API docs). Ignore invalid filters.
        self.ec2_instance_filters = defaultdict(list)
        if config.has_option('ec2', 'instance_filters'):
            for instance_filter in config.get('ec2', 'instance_filters', '').split(','):
                instance_filter = instance_filter.strip()
                if not instance_filter or '=' not in instance_filter:
                    continue
                filter_key, filter_value = [x.strip() for x in instance_filter.split('=', 1)]
                if not filter_key:
                    continue
                self.ec2_instance_filters[filter_key].append(filter_value)