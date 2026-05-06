def download(self, job_id, files, **kwargs):
        """Downloads files from server.
        """
        check_jobid(job_id)

        if not files:
            return
        if isinstance(files, string_types):
            files = [files]
        directory = False
        recursive = kwargs.pop('recursive', True)

        if 'destination' in kwargs and 'directory' in kwargs:
            raise TypeError("Only use one of 'destination' or 'directory'")
        elif 'destination' in kwargs:
            destination = Path(kwargs.pop('destination'))
            if len(files) != 1:
                raise ValueError("'destination' specified but multiple files "
                                 "given; did you mean to use 'directory'?")
        elif 'directory' in kwargs:
            destination = Path(kwargs.pop('directory'))
            directory = True
        if kwargs:
            raise TypeError("Got unexpected keyword arguments")

        # Might raise JobNotFound
        status, target, result = self.status(job_id)

        scp_client = self.get_scp_client()
        for filename in files:
            logger.info("Downloading %s", target / filename)
            if directory:
                scp_client.get(str(target / filename),
                               str(destination / filename),
                               recursive=recursive)
            else:
                scp_client.get(str(target / filename),
                               str(destination),
                               recursive=recursive)