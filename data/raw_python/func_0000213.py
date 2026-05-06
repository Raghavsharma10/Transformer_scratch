def _build_mappings(
        self, classes: Sequence[type]
    ) -> Tuple[Mapping[type, Sequence[type]], Mapping[type, Sequence[type]]]:
        """
        Collect all bases and organize into parent/child mappings.
        """
        parents_to_children: MutableMapping[type, Set[type]] = {}
        children_to_parents: MutableMapping[type, Set[type]] = {}
        visited_classes: Set[type] = set()
        class_stack = list(classes)
        while class_stack:
            class_ = class_stack.pop()
            if class_ in visited_classes:
                continue
            visited_classes.add(class_)
            for base in class_.__bases__:
                if base not in visited_classes:
                    class_stack.append(base)
                parents_to_children.setdefault(base, set()).add(class_)
                children_to_parents.setdefault(class_, set()).add(base)
        sorted_parents_to_children: MutableMapping[
            type, List[type]
        ] = collections.OrderedDict()
        for parent, children in sorted(
            parents_to_children.items(), key=lambda x: (x[0].__module__, x[0].__name__)
        ):
            sorted_parents_to_children[parent] = sorted(
                children, key=lambda x: (x.__module__, x.__name__)
            )
        sorted_children_to_parents: MutableMapping[
            type, List[type]
        ] = collections.OrderedDict()
        for child, parents in sorted(
            children_to_parents.items(), key=lambda x: (x[0].__module__, x[0].__name__)
        ):
            sorted_children_to_parents[child] = sorted(
                parents, key=lambda x: (x.__module__, x.__name__)
            )
        return sorted_parents_to_children, sorted_children_to_parents