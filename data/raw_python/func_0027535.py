def add_element(self, name, ns_uri=None, attributes=None,
            text=None, before_this_element=False):
        """
        Add a new child element to this element, with an optional namespace
        definition. If no namespace is provided the child will be assigned
        to the default namespace.

        :param string name: a name for the child node. The name may be used
            to apply a namespace to the child by including:

            - a prefix component in the name of the form
              ``ns_prefix:element_name``, where the prefix has already been
              defined for a namespace URI (such as via :meth:`set_ns_prefix`).
            - a literal namespace URI value delimited by curly braces, of
              the form ``{ns_uri}element_name``.
        :param ns_uri: a URI specifying the new element's namespace. If the
            ``name`` parameter specifies a namespace this parameter is ignored.
        :type ns_uri: string or None
        :param attributes: collection of attributes to assign to the new child.
        :type attributes: dict, list, tuple, or None
        :param text: text value to assign to the new child.
        :type text: string or None
        :param bool before_this_element: if *True* the new element is
            added as a sibling preceding this element, instead of as a child.
            In other words, the new element will be a child of this element's
            parent node, and will immediately precent this element in the DOM.

        :return: the new child as a an :class:`Element` node.
        """
        # Determine local name, namespace and prefix info from tag name
        prefix, local_name, node_ns_uri = \
            self.adapter.get_ns_info_from_node_name(name, self.impl_node)
        if prefix:
            qname = u'%s:%s' % (prefix, local_name)
        else:
            qname = local_name
        # If no name-derived namespace, apply an alternate namespace
        if node_ns_uri is None:
            if ns_uri is None:
                # Default document namespace
                node_ns_uri = self.adapter.get_ns_uri_for_prefix(
                    self.impl_node, None)
            else:
                # keyword-parameter namespace
                node_ns_uri = ns_uri
        # Create element
        child_elem = self.adapter.new_impl_element(
            qname, node_ns_uri, parent=self.impl_node)
        # If element's default namespace was defined by literal uri prefix,
        # create corresponding xmlns attribute for element...
        if not prefix and '}' in name:
            self._set_element_attributes(child_elem,
                {'xmlns': node_ns_uri}, ns_uri=self.XMLNS_URI)
        # ...otherwise define keyword-defined namespace as the default, if any
        elif ns_uri is not None:
            self._set_element_attributes(child_elem,
                {'xmlns': ns_uri}, ns_uri=self.XMLNS_URI)
        # Create subordinate nodes
        if attributes is not None:
            self._set_element_attributes(child_elem, attr_obj=attributes)
        if text is not None:
            self._add_text(child_elem, text)
        # Add new element to its parent before a given node...
        if before_this_element:
            self.adapter.add_node_child(
                self.adapter.get_node_parent(self.impl_node),
                child_elem, before_sibling=self.impl_node)
        # ...or in the default position, appended after existing nodes
        else:
            self.adapter.add_node_child(self.impl_node, child_elem)
        return self.adapter.wrap_node(
            child_elem, self.adapter.impl_document, self.adapter)