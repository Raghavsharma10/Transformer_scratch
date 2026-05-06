def fromFile(cls, person, inputFile, format):
        """
        Create a L{Mugshot} item for C{person} out of the image data in
        C{inputFile}, or update C{person}'s existing L{Mugshot} item to
        reflect the new images.

        @param inputFile: An image of a person.
        @type inputFile: C{file}

        @param person: The person this mugshot is to be associated with.
        @type person: L{Person}

        @param format: The format of the data in C{inputFile}.
        @type format: C{unicode} (e.g. I{jpeg})

        @rtype: L{Mugshot}
        """
        body = cls.makeThumbnail(inputFile, person, format, smaller=False)
        inputFile.seek(0)
        smallerBody = cls.makeThumbnail(
            inputFile, person, format, smaller=True)

        ctype = u'image/' + format

        self = person.store.findUnique(
            cls, cls.person == person, default=None)
        if self is None:
            self = cls(store=person.store,
                       person=person,
                       type=ctype,
                       body=body,
                       smallerBody=smallerBody)
        else:
            self.body = body
            self.smallerBody = smallerBody
            self.type = ctype
        return self