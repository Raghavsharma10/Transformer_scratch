def sync_config(self, force=False):
        """Sync the file config into the library proxy data in the root dataset """
        from ambry.library.config import LibraryConfigSyncProxy
        lcsp = LibraryConfigSyncProxy(self)
        lcsp.sync(force=force)