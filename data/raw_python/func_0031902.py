def reload_glance(self, target_app, slices=None):
        """
        Reloads an app's glance. Blocks as long as necessary.

        :param target_app: The UUID of the app for which to reload its glance.
        :type target_app: ~uuid.UUID
        :param slices: The slices with which to reload the app's glance.
        :type slices: list[.AppGlanceSlice]
        """
        glance = AppGlance(
            version=1,
            creation_time=time.time(),
            slices=(slices or [])
        )
        SyncWrapper(self._blobdb.insert, BlobDatabaseID.AppGlance, target_app, glance.serialise()).wait()