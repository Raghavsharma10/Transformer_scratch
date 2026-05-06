def add_subject(self, subject_id, attributes, parents=[],
            issuer='default'):
        """
        Will add the given subject with a given identifier and attribute
        dictionary.

            example/

                add_subject('/user/j12y', {'username': 'j12y'})
        """
        # MAINT: consider test to avoid adding duplicate subject id
        assert isinstance(attributes, (dict)), "attributes expected to be dict"

        attrs = []
        for key in attributes.keys():
            attrs.append({
                'issuer': issuer,
                'name': key,
                'value': attributes[key]
                })

        body = {
            "subjectIdentifier": subject_id,
            "parents": parents,
            "attributes": attrs,
        }

        return self._put_subject(subject_id, body)