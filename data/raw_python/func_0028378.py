async def upload(self, files: Sequence[Union[str, Path]],
                     basedir: Union[str, Path] = None,
                     show_progress: bool = False):
        '''
        Uploads the given list of files to the compute session.
        You may refer them in the batch-mode execution or from the code
        executed in the server afterwards.

        :param files: The list of file paths in the client-side.
            If the paths include directories, the location of them in the compute
            session is calculated from the relative path to *basedir* and all
            intermediate parent directories are automatically created if not exists.

            For example, if a file path is ``/home/user/test/data.txt`` (or
            ``test/data.txt``) where *basedir* is ``/home/user`` (or the current
            working directory is ``/home/user``), the uploaded file is located at
            ``/home/work/test/data.txt`` in the compute session container.
        :param basedir: The directory prefix where the files reside.
            The default value is the current working directory.
        :param show_progress: Displays a progress bar during uploads.
        '''
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
        base_path = (Path.cwd() if basedir is None
                     else Path(basedir).resolve())
        files = [Path(file).resolve() for file in files]
        total_size = 0
        for file_path in files:
            total_size += file_path.stat().st_size
        tqdm_obj = tqdm(desc='Uploading files',
                        unit='bytes', unit_scale=True,
                        total=total_size,
                        disable=not show_progress)
        with tqdm_obj:
            attachments = []
            for file_path in files:
                try:
                    attachments.append(AttachedFile(
                        str(file_path.relative_to(base_path)),
                        ProgressReportingReader(str(file_path),
                                                tqdm_instance=tqdm_obj),
                        'application/octet-stream',
                    ))
                except ValueError:
                    msg = 'File "{0}" is outside of the base directory "{1}".' \
                          .format(file_path, base_path)
                    raise ValueError(msg) from None

            rqst = Request(self.session,
                           'POST', '/kernel/{}/upload'.format(self.kernel_id),
                           params=params)
            rqst.attach_files(attachments)
            async with rqst.fetch() as resp:
                return resp