def expand_db_attributes(attrs, for_editor):
        """
        Given a dictionary of attributes, find the corresponding link instance and
        return its HTML representation.

        :param attrs: dictionary of link attributes.
        :param for_editor: whether or not HTML is for editor.
        :rtype: str.
        """
        try:
            editor_attrs    = ''
            link            = Link.objects.get(id=attrs['id'])

            if for_editor:
                editor_attrs = 'data-linktype="link" data-id="{0}" '.format(
                    link.id
                )

            return '<a {0}href="{1}" title="{2}">'.format(
                editor_attrs,
                escape(link.get_absolute_url()),
                link.title
            )
        except Link.DoesNotExist:
            return '<a>'