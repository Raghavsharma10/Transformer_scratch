def createSQL(self, sql, args=()):
        """
        For use with auto-committing statements such as CREATE TABLE or CREATE
        INDEX.
        """
        before = time.time()
        self._execSQL(sql, args)
        after = time.time()
        if after - before > 2.0:
            log.msg('Extremely long CREATE: %s' % (after - before,))
            log.msg(sql)