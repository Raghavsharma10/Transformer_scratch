async def download(self, files: Sequence[Union[str, Path]],
                       dest: Union[str, Path] = '.',
                       show_progress: bool = False):
        '''
        Downloads the given list of files from the compute session.

        :param files: The list of file paths in the compute session.
            If they are relative paths, the path is calculated from
            ``/home/work`` in the compute session container.
        :param dest: The destination directory in the client-side.
        :param show_progress: Displays a progress bar during downloads.
        '''
        params = {}
        if self.owner_access_key:
            params['owner_access_key'] = self.owner_access_key
        rqst = Request(self.session,
                       'GET', '/kernel/{}/download'.format(self.kernel_id),
                       params=params)
        rqst.set_json({
            'files': [*map(str, files)],
        })
        async with rqst.fetch() as resp:
            chunk_size = 1 * 1024
            file_names = None
            tqdm_obj = tqdm(desc='Downloading files',
                            unit='bytes', unit_scale=True,
                            total=resp.content.total_bytes,
                            disable=not show_progress)
            with tqdm_obj as pbar:
                fp = None
                while True:
                    chunk = await resp.aread(chunk_size)
                    if not chunk:
                        break
                    pbar.update(len(chunk))
                    # TODO: more elegant parsing of multipart response?
                    for part in chunk.split(b'\r\n'):
                        if part.startswith(b'--'):
                            if fp:
                                fp.close()
                                with tarfile.open(fp.name) as tarf:
                                    tarf.extractall(path=dest)
                                    file_names = tarf.getnames()
                                os.unlink(fp.name)
                            fp = tempfile.NamedTemporaryFile(suffix='.tar',
                                                             delete=False)
                        elif part.startswith(b'Content-') or part == b'':
                            continue
                        else:
                            fp.write(part)
                if fp:
                    fp.close()
                    os.unlink(fp.name)
            result = {'file_names': file_names}
            return result