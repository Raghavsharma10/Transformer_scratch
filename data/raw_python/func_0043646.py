def load_children(self):
        """
        Load the subelements from the xml_element in its correspondent classes.

        :returns: List of child objects.
        :rtype: list
        :raises CardinalityException: If there is more than one Version child.
        :raises CardinalityException: If there is no Version child.
        :raises CardinalityException: If there is no Profile element.
        """
        # Containers
        children = list()
        statuses = list()
        version = None
        profiles = list()

        # Element load
        for element in self.xml_element:
            uri, tag = Element.get_namespace_and_tag(element.tag)
            if tag == 'version':
                if version is None:
                    version = TailoringVersion(element)
                else:
                    error_msg = 'version element found more than once'
                    raise CardinalityException(error_msg)
            elif tag == 'status':
                statuses.append(Status(element))
            elif tag == 'Profile':
                profiles.append(Profile(element))

        # Element validation
        if version is None:
            error_msg = 'version element is required'
            raise CardinalityException(error_msg)
        if len(profiles) <= 0:
            error_msg = 'Profile element is required at least once'
            raise CardinalityException(error_msg)

        # List construction
        children.extend(statuses)
        if version is not None:
            children.append(version)
        children.extend(profiles)

        return children