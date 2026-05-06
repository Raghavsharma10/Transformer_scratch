def get_contrib_names(self, contrib):
        """
        Returns an appropriate Name and File-As-Name for a contrib element.

        This code was refactored out of nav_contributors and
        package_contributors to provide a single definition point for a common
        job. This is a useful utility that may be well-employed for other
        publishers as well.
        """
        collab = contrib.find('collab')
        anon = contrib.find('anonymous')
        if collab is not None:
            proper_name = serialize(collab, strip=True)
            file_as_name = proper_name
        elif anon is not None:
            proper_name = 'Anonymous'
            file_as_name = proper_name
        else:
            name = contrib.find('name')
            surname = name.find('surname').text
            given = name.find('given-names')
            if given is not None:
                if given.text:  # Sometimes these tags are empty
                    proper_name = ' '.join([surname, given.text])
                    #File-as name is <surname>, <given-initial-char>
                    file_as_name = ', '.join([surname, given.text[0]])
                else:
                    proper_name = surname
                    file_as_name = proper_name
            else:
                proper_name = surname
                file_as_name = proper_name
        return proper_name, file_as_name