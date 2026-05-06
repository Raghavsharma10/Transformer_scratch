def title(self):
        """
        The title of the course. If no entry in the namemap of the configuration is found a new entry is created with name=$STUD.IP_NAME + $SEMESTER_NAME
        """
        name = c.namemap_lookup(self.id)
        if name is None:
            name = self._title + " " + client.get_semester_title(self)
            c.namemap_set(self.id, name)
        return secure_filename(name)