def placeholderForPerson(cls, person):
        """
        Make an unstored, placeholder L{Mugshot} instance for the given
        person.

        @param person: A person without a L{Mugshot}.
        @type person: L{Person}

        @rtype: L{Mugshot}
        """
        imageDir = FilePath(__file__).parent().child(
            'static').child('images')
        return cls(
            type=u'image/png',
            body=imageDir.child('mugshot-placeholder.png'),
            smallerBody=imageDir.child(
                'mugshot-placeholder-smaller.png'),
            person=person)