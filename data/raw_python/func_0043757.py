def load_children(self):
        """
        Load the subelements from the xml_element in its correspondent classes.

        :returns: List of child objects.
        :rtype: list
        :raises CardinalityException: If there is more than one Version child.
        """

        # Containers
        children = list()
        statuses = list()
        version = None
        titles = list()
        descriptions = list()
        platforms = list()
        idents = list()

        # Element load
        for element in self.xml_element:
            uri, tag = Element.get_namespace_and_tag(element.tag)
            if tag == 'version':
                if version is None:
                    version = Version(element)
                else:
                    error_msg = 'version element found more than once'
                    raise CardinalityException(error_msg)
            elif tag == 'status':
                statuses.append(Status(element))
            elif tag == 'title':
                titles.append(Title(element))
            elif tag == 'description':
                descriptions.append(Description(element))
            elif tag == 'platform':
                platforms.append(Platform(element))
            elif tag == 'ident':
                idents.append(Ident(element))

        # List construction
        children.extend(statuses)
        if version is not None:
            children.append(version)
        children.extend(titles)
        children.extend(descriptions)
        children.extend(platforms)
        children.extend(idents)

        return children