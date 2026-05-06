def _generate_service(service_config):
        '''
        Generate a service from a service_config dictionary

        Parameters
        ----------

        service_config : dict
            Configuration with keys service, args, and
            kwargs used to generate a new fs service
            object

        Returns
        -------

        service : object
            fs service object initialized with *args,
            **kwargs

        Examples
        --------

        Generate a temporary filesystem (no arguments
        required):

        .. code-block:: python

            >>> tmp = APIConstructor._generate_service(
            ...     {'service': 'TempFS'})
            ...
            >>> from fs.tempfs import TempFS
            >>> assert isinstance(tmp, TempFS)
            >>> import os
            >>> assert os.path.isdir(tmp.getsyspath('/'))
            >>> tmp.close()

        Generate a system filesystem in a temporary
        directory:

        .. code-block:: python

            >>> import tempfile
            >>> tempdir = tempfile.mkdtemp()
            >>> local = APIConstructor._generate_service(
            ...     {
            ...         'service': 'OSFS',
            ...         'args': [tempdir]
            ...     })
            ...
            >>> from fs.osfs import OSFS
            >>> assert isinstance(local, OSFS)
            >>> import os
            >>> assert os.path.isdir(local.getsyspath('/'))
            >>> local.close()
            >>> import shutil
            >>> shutil.rmtree(tempdir)

        Mock an S3 filesystem with moto:

        .. code-block:: python

            >>> import moto
            >>> m = moto.mock_s3()
            >>> m.start()
            >>> s3 = APIConstructor._generate_service(
            ...     {
            ...         'service': 'S3FS',
            ...         'args': ['bucket-name'],
            ...         'kwargs': {
            ...             'aws_access_key':'MY_KEY',
            ...             'aws_secret_key':'MY_SECRET_KEY'
            ...         }
            ...     })
            ...
            >>> from fs.s3fs import S3FS
            >>> assert isinstance(s3, S3FS)
            >>> m.stop()

        '''

        filesystems = []

        for _, modname, _ in pkgutil.iter_modules(fs.__path__):
            if modname.endswith('fs'):
                filesystems.append(modname)

        service_mod_name = service_config['service'].lower()

        assert_msg = 'Filesystem "{}" not found in pyFilesystem {}'.format(
            service_mod_name, fs.__version__)

        assert service_mod_name in filesystems, assert_msg

        svc_module = importlib.import_module('fs.{}'.format(service_mod_name))
        svc_class = svc_module.__dict__[service_config['service']]

        service = svc_class(*service_config.get('args', []),
                            **service_config.get('kwargs', {}))

        return service