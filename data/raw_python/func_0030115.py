def _remote(self, name):
        """Return a remote for which 'name' matches the short_name or url """
        from ambry.orm import Remote
        from sqlalchemy import or_
        from ambry.orm.exc import NotFoundError
        from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound

        if not name.strip():
            raise NotFoundError("Empty remote name")

        try:
            try:
                r = self.database.session.query(Remote).filter(Remote.short_name == name).one()
            except NoResultFound as e:
                r = None

            if not r:
                r = self.database.session.query(Remote).filter(Remote.url == name).one()

        except NoResultFound as e:
            raise NotFoundError(str(e)+'; '+name)
        except MultipleResultsFound as e:
            self.logger.error("Got multiple results for search for remote '{}': {}".format(name, e))
            return None

        return r