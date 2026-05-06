def download_to_path(self, gsuri, localpath, binary_mode=False, tmpdir=None):
        """
        This method is analogous to "gsutil cp gsuri localpath", but in a
        programatically accesible way. The only difference is that we
        have to make a guess about the encoding of the file to not upset
        downstream file operations. If you are downloading a VCF, then
        "False" is great. If this is a BAM file you are asking for, you
        should enable the "binary_mode" to make sure file doesn't get
        corrupted.

        gsuri: full GS-based URI, e.g. gs://cohorts/rocks.txt
        localpath: the path for the downloaded file, e.g. /mnt/cohorts/yep.txt
        binary_mode: (logical) if yes, the binary file operations will be
                     used; if not, standard ascii-based ones.
        """
        bucket_name, gs_rel_path = self.parse_uri(gsuri)
        # And now request the handles for bucket and the file
        bucket = self._client.get_bucket(bucket_name)
        # Just assignment, no downloading (yet)
        ablob = bucket.get_blob(gs_rel_path)
        if not ablob:
            raise GoogleStorageIOError(
                "No such file on Google Storage: '{}'".format(gs_rel_path))

        # A tmp file to serve intermediate phase
        # should be on same filesystem as localpath
        tmp_fid, tmp_file_path = tempfile.mkstemp(text=(not binary_mode),
                                                  dir=tmpdir)
        # set chunk_size to reasonable default
        # https://github.com/GoogleCloudPlatform/google-cloud-python/issues/2222
        ablob.chunk_size = 1<<30
        # Download starts in a sec....
        ablob.download_to_filename(client=self._client, filename=tmp_file_path)
        # ... end download ends. Let's move our finished file over.

        # You will see that below, instead of directly writing to a file
        # we are instead first using a different file and then move it to
        # its final location. We are doing this because we don't want
        # corrupted/incomplete data to be around as much as possible.
        return os.rename(tmp_file_path, localpath)