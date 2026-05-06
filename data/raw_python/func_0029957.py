def log_pipeline(self, pl):
        """Write a report of the pipeline out to a file """
        from datetime import datetime
        from ambry.etl.pipeline import CastColumns

        self.build_fs.makedir('pipeline', allow_recreate=True)

        try:
            ccp = pl[CastColumns]
            caster_code = ccp.pretty_code
        except Exception as e:
            caster_code = str(e)

        templ = u("""
Pipeline     : {}
run time     : {}
phase        : {}
source name  : {}
source table : {}
dest table   : {}
========================================================
{}

Pipeline Headers
================
{}

Caster Code
===========
{}

""")
        try:
            v = templ.format(pl.name, str(datetime.now()), pl.phase, pl.source_name, pl.source_table,
                             pl.dest_table, unicode(pl), pl.headers_report(), caster_code)
        except UnicodeError as e:
            v = ''
            self.error('Faled to write pipeline log for pipeline {} '.format(pl.name))

        path = os.path.join('pipeline', pl.phase + '-' + pl.file_name + '.txt')

        self.build_fs.makedir(os.path.dirname(path), allow_recreate=True, recursive=True)
        # LazyFS should handled differently because of:
        # TypeError: lazy_fs.setcontents(..., encoding='utf-8') got an unexpected keyword argument 'encoding'
        if isinstance(self.build_fs, LazyFS):
            self.build_fs.wrapped_fs.setcontents(path, v, encoding='utf8')
        else:
            self.build_fs.setcontents(path, v, encoding='utf8')