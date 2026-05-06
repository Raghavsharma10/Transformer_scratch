def create_pool(self, name, raid_groups, description=None, **kwargs):
        """Create pool based on RaidGroupParameter.

        :param name: pool name
        :param raid_groups: a list of *RaidGroupParameter*
        :param description: pool description
        :param alert_threshold: Threshold at which the system will generate
               alerts about the free space in the pool, specified as
               a percentage.
        :param is_harvest_enabled:
               True - Enable pool harvesting for the pool.
               False - Disable pool harvesting for the pool.
        :param is_snap_harvest_enabled:
               True - Enable snapshot harvesting for the pool.
               False - Disable snapshot harvesting for the pool.
        :param pool_harvest_high_threshold: Pool used space high threshold at
               which the system will automatically starts to delete snapshots
               in the pool
        :param pool_harvest_low_threshold: Pool used space low threshold under
               which the system will automatically stop deletion of snapshots
               in the pool
        :param snap_harvest_high_threshold: Snapshot used space high threshold
               at which the system automatically starts to delete snapshots
               in the pool
        :param snap_harvest_low_threshold: Snapshot used space low threshold
               below which the system will stop automatically deleting
               snapshots in the pool
        :param is_fast_cache_enabled:
               True - FAST Cache will be enabled for this pool.
               False - FAST Cache will be disabled for this pool.
        :param is_fastvp_enabled:
               True - Enable scheduled data relocations for the pool.
               False - Disable scheduled data relocations for the pool.
        :param pool_type:
               StoragePoolTypeEnum.TRADITIONAL - Create traditional pool.
               StoragePoolTypeEnum.DYNAMIC - Create dynamic pool. (default)
        """
        return UnityPool.create(self._cli, name=name, description=description,
                                raid_groups=raid_groups, **kwargs)