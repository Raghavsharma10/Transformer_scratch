def session(self):
        """Return a SqlAlchemy session."""
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.event import listen

        if not self.Session:
            self.Session = sessionmaker(bind=self.engine)

        if not self._session:
            self._session = self.Session()
            # set the search path

            if self._schema:
                def after_begin(session, transaction, connection):
                    # import traceback
                    # print traceback.print_stack()
                    session.execute('SET search_path TO {}'.format(self._schema))

                listen(self._session, 'after_begin', after_begin)

        return self._session