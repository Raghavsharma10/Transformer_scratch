def randomSlugField(self):
        """
        Return the unique slug by generating the uuid4
        to fix the duplicate slug (unique=True)
        """
        lst = [
            "sample-slug-{}".format(uuid.uuid4().hex),
            "awesome-djipsum-{}".format(uuid.uuid4().hex),
            "unique-slug-{}".format(uuid.uuid4().hex)
        ]
        return self.randomize(lst)