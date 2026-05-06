def build_urls(self: NodeVisitor, node: inheritance_diagram) -> Mapping[str, str]:
    """
    Builds a mapping of class paths to URLs.
    """
    current_filename = self.builder.current_docname + self.builder.out_suffix
    urls = {}
    for child in node:
        # Another document
        if child.get("refuri") is not None:
            uri = child.get("refuri")
            package_path = child["reftitle"]
            if uri.startswith("http"):
                _, _, package_path = uri.partition("#")
            else:
                uri = (
                    pathlib.Path("..")
                    / pathlib.Path(current_filename).parent
                    / pathlib.Path(uri)
                )
                uri = str(uri).replace(os.path.sep, "/")
            urls[package_path] = uri
        # Same document
        elif child.get("refid") is not None:
            urls[child["reftitle"]] = (
                "../" + current_filename + "#" + child.get("refid")
            )
    return urls