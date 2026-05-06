def clearAnnouncements(self):
        """
        Flushes the announcement table.
        """
        try:
            q = models.Announcement.delete().where(
                models.Announcement.id > 0)
            q.execute()
        except Exception as e:
            raise exceptions.RepoManagerException(e)