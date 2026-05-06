def to_embedded(pool_id=None, is_thin_enabled=None,
                    is_deduplication_enabled=None, is_compression_enabled=None,
                    is_backup_only=None, size=None, tiering_policy=None,
                    request_id=None, src_id=None, name=None, default_sp=None,
                    replication_resource_type=None):
        """
        Constructs an embeded object of `UnityResourceConfig`.

        :param pool_id: storage pool of the resource.
        :param is_thin_enabled: is thin type or not.
        :param is_deduplication_enabled: is deduplication enabled or not.
        :param is_compression_enabled: is in-line compression (ILC) enabled or
            not.
        :param is_backup_only: is backup only or not.
        :param size: size of the resource.
        :param tiering_policy: `TieringPolicyEnum` value. Tiering policy
            for the resource.
        :param request_id: unique request ID for the configuration.
        :param src_id: storage resource if it already exists.
        :param name: name of the storage resource.
        :param default_sp: `NodeEnum` value. Default storage processor for
            the resource.
        :param replication_resource_type: `ReplicationEndpointResourceTypeEnum`
            value. Replication resource type.
        :return:
        """
        return {'poolId': pool_id, 'isThinEnabled': is_thin_enabled,
                'isDeduplicationEnabled': is_deduplication_enabled,
                'isCompressionEnabled': is_compression_enabled,
                'isBackupOnly': is_backup_only, 'size': size,
                'tieringPolicy': tiering_policy, 'requestId': request_id,
                'srcId': src_id, 'name': name, 'defaultSP': default_sp,
                'replicationResourceType': replication_resource_type}