def __display_for_tree_node(self, tree_node):
        """ Return the text display for the given tree node. Based on number of keys associated with tree node. """
        keys = tree_node.keys
        if len(keys) == 1:
            return "{0} ({1})".format(tree_node.keys[-1], tree_node.count)
        elif len(keys) == 2:
            months = (_("January"), _("February"), _("March"), _("April"), _("May"), _("June"), _("July"), _("August"),
                      _("September"), _("October"), _("November"), _("December"))
            return "{0} ({1})".format(months[max(min(tree_node.keys[1]-1, 11), 0)], tree_node.count)
        else:
            weekdays = (_("Monday"), _("Tuesday"), _("Wednesday"), _("Thursday"), _("Friday"), _("Saturday"), _("Sunday"))
            date = datetime.date(tree_node.keys[0], tree_node.keys[1], tree_node.keys[2])
            return "{0} - {1} ({2})".format(tree_node.keys[2], weekdays[date.weekday()], tree_node.count)