def emit(self, record):
        """ Handle the logging call. """

        if bool(self.store_only):
            do_not_store = True
            for extra in self.store_only:
                if bool(getattr(record, extra, None)):
                    do_not_store = False
                    break

            if do_not_store:
                return

        self.db = sqlite3.connect(self.filename)

        self.db.execute(
            """
                INSERT INTO log(
                    log_level,
                    log_level_name,

                    name,
                    message,
                    args,

                    module,
                    func_name,
                    line_no,
                    filename,

                    exception,
                    process,
                    thread,
                    thread_name,

                    siteconfig
                )
                VALUES(
                    ?,?,
                    ?,?,?,
                    ?,?,?,?,
                    ?,?,?,?,
                    ?
                );""",
            (
                record.levelno,
                record.levelname,

                record.name,
                record.msg,
                json.dumps(record.args, cls=OrderedEncoder),

                record.module,
                record.funcName,
                record.lineno,
                os.path.abspath(record.filename),

                record.exc_text,
                record.process,
                record.thread,
                record.threadName,

                getattr(record, 'siteconfig', None),
            )
        )
        self.db.commit()
        self.db.close()