def forwards(self, orm):
        "Perform a 'safe' load using Avocado's backup utilities."
        from avocado.core import backup
        backup.safe_load(u'0002_avocado_metadata', backup_path=None,
            using='default')