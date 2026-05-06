def _get_recursive_dependancies(self, dependencies_map, sourcepath,
                                    recursive=True):
        """
        Return all dependencies of a source, recursively searching through its
        dependencies.

        This is a common method used by ``children`` and ``parents`` methods.

        Args:
            dependencies_map (dict): Internal buffer (internal buffers
                ``_CHILDREN_MAP`` or ``_PARENTS_MAP``) to use for searching.
            sourcepath (str): Source file path to start searching for
                dependencies.

        Keyword Arguments:
            recursive (bool): Switch to enable recursive finding (if True).
                Default to True.

        Raises:
            CircularImport: If circular error is detected from a source.

        Returns:
            set: List of dependencies paths.
        """
        # Direct dependencies
        collected = set([])
        collected.update(dependencies_map.get(sourcepath, []))

        # Sequence of 'dependencies_map' items to explore
        sequence = collected.copy()
        # Exploration list
        walkthrough = []

        # Recursive search starting from direct dependencies
        if recursive:
            while True:
                if not sequence:
                    break
                item = sequence.pop()

                # Add current source to the explorated source list
                walkthrough.append(item)

                # Current item children
                current_item_dependancies = dependencies_map.get(item, [])

                for dependency in current_item_dependancies:
                    # Allready visited item, ignore and continue to the new
                    # item
                    if dependency in walkthrough:
                        continue
                    # Unvisited item yet, add its children to dependencies and
                    # item to explore
                    else:
                        collected.add(dependency)
                        sequence.add(dependency)

                # Sourcepath has allready been visited but present itself
                # again, assume it's a circular import
                if sourcepath in walkthrough:
                    msg = "A circular import has occured by '{}'"
                    raise CircularImport(msg.format(current_item_dependancies))

                # No more item to explore, break loop
                if not sequence:
                    break

        return collected