def education(self):
        """
        A list of structures describing the user's education history.

        Each structure has attributes ``school``, ``year``, ``concentration`` and ``type``.

        ``school``, ``year`` reference ``Page`` instances, while ``concentration`` is a list of ``Page``
        instances. ``type`` is just a string that describes the education level.

        .. note:: ``concentration`` may be ``False`` if the user has not specified his/her
                  concentration for the given school.
        """
        educations = []

        for education in self.cache['education']:
            school        = Page(**education.get('school'))
            year          = Page(**education.get('year'))
            type          = education.get('type')
            
            if 'concentration' in education:
                concentration = map(lambda c: Page(**c), education.get('concentration'))
            else:
                concentration = False

            education = Structure(
                school = school,
                year = year,
                concentration = concentration,
                type = type
            )

            educations.append(education)

        return educations