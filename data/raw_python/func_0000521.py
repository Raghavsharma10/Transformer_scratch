def build_node_tree(self, source_paths):
        """
        Build a node tree.
        """
        import uqbar.apis

        root = PackageNode()
        # Build node tree, top-down
        for source_path in sorted(
            source_paths, key=lambda x: uqbar.apis.source_path_to_package_path(x)
        ):
            package_path = uqbar.apis.source_path_to_package_path(source_path)
            parts = package_path.split(".")
            if not self.document_private_modules and any(
                part.startswith("_") for part in parts
            ):
                continue
            # Find parent node.
            parent_node = root
            if len(parts) > 1:
                parent_package_path = ".".join(parts[:-1])
                try:
                    parent_node = root[parent_package_path]
                except KeyError:
                    parent_node = root
                try:
                    if parent_node is root:
                        # Backfill missing parent node.
                        grandparent_node = root
                        if len(parts) > 2:
                            grandparent_node = root[
                                parent_package_path.rpartition(".")[0]
                            ]
                        parent_node = PackageNode(name=parent_package_path)
                        grandparent_node.append(parent_node)
                        grandparent_node[:] = sorted(
                            grandparent_node, key=lambda x: x.package_path
                        )
                except KeyError:
                    parent_node = root
            # Create or update child node.
            node_class = ModuleNode
            if source_path.name == "__init__.py":
                node_class = PackageNode
            try:
                # If the child exists, it was previously backfilled.
                child_node = root[package_path]
                child_node.source_path = source_path
            except KeyError:
                # Otherwise it needs to be created and appended to the parent.
                child_node = node_class(name=package_path, source_path=source_path)
                parent_node.append(child_node)
                parent_node[:] = sorted(parent_node, key=lambda x: x.package_path)
        # Build documenters, bottom-up.
        # This allows parent documenters to easily aggregate their children.
        for node in root.depth_first(top_down=False):
            kwargs = dict(
                document_private_members=self.document_private_members,
                member_documenter_classes=self.member_documenter_classes,
            )
            if isinstance(node, ModuleNode):
                node.documenter = self.module_documenter_class(
                    node.package_path, **kwargs
                )
            else:
                # Collect references to child modules and packages.
                node.documenter = self.module_documenter_class(
                    node.package_path,
                    module_documenters=[
                        child.documenter
                        for child in node
                        if child.documenter is not None
                    ],
                    **kwargs,
                )
            if (
                not self.document_empty_modules
                and not node.documenter.module_documenters
                and not node.documenter.member_documenters
            ):
                node.parent.remove(node)
        return root