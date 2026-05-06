def keywords(self) -> Set[str]:
        """A set of all keywords of all handled devices.

        In addition to attribute access via device names, |Nodes| and
        |Elements| objects allow for attribute access via keywords,
        allowing for an efficient search of certain groups of devices.
        Let us use the example from above, where the nodes `na` and `nb`
        have no keywords, but each of the other three nodes both belongs
        to either `group_a` or `group_b` and `group_1` or `group_2`:

        >>> from hydpy import Node, Nodes
        >>> nodes = Nodes('na',
        ...               Node('nb', variable='W'),
        ...               Node('nc', keywords=('group_a', 'group_1')),
        ...               Node('nd', keywords=('group_a', 'group_2')),
        ...               Node('ne', keywords=('group_b', 'group_1')))
        >>> nodes
        Nodes("na", "nb", "nc", "nd", "ne")
        >>> sorted(nodes.keywords)
        ['group_1', 'group_2', 'group_a', 'group_b']

        If you are interested in inspecting all devices belonging to
        `group_a`, select them via this keyword:

        >>> subgroup = nodes.group_1
        >>> subgroup
        Nodes("nc", "ne")

        You can further restrict the search by also selecting the devices
        belonging to `group_b`, which holds only for node "e", in the given
        example:

        >>> subsubgroup = subgroup.group_b
        >>> subsubgroup
        Node("ne", variable="Q",
             keywords=["group_1", "group_b"])

        Note that the keywords already used for building a device subgroup
        are not informative anymore (as they hold for each device) and are
        thus not shown anymore:

        >>> sorted(subgroup.keywords)
        ['group_a', 'group_b']

        The latter might be confusing if you intend to work with a device
        subgroup for a longer time.  After copying the subgroup, all
        keywords of the contained devices are available again:

        >>> from copy import copy
        >>> newgroup = copy(subgroup)
        >>> sorted(newgroup.keywords)
        ['group_1', 'group_a', 'group_b']
        """
        return set(keyword for device in self
                   for keyword in device.keywords if
                   keyword not in self._shadowed_keywords)