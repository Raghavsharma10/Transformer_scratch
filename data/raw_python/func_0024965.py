def create_blobstore(self, **kwargs):
        """
        Creates an instance of the BlobStore Service.
        """
        blobstore = predix.admin.blobstore.BlobStore(**kwargs)
        blobstore.create()

        blobstore.add_to_manifest(self)
        return blobstore