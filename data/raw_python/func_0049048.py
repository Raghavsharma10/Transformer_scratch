def get_parent_book_nodes(self):
        """Gets the parents of this book.

        return: (osid.commenting.BookNodeList) - the parents of this
                book
        *compliance: mandatory -- This method must be implemented.*

        """
        parent_book_nodes = []
        for node in self._my_map['parentNodes']:
            parent_book_nodes.append(BookNode(
                node._my_map,
                runtime=self._runtime,
                proxy=self._proxy,
                lookup_session=self._lookup_session))
        return BookNodeList(parent_book_nodes)