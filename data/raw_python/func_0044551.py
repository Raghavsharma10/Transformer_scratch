def get_canonical_uri_path(uri_path):
    """
    get_canonical_uri_path(uri_path) -> str

    Normalizes the specified URI path component, removing redundant slashes
    and relative path components.

    A ValueError exception is raised if:
    * The URI path is not empty and not absolute (does not start with '/').
    * A parent relative path element ('..') attempts to go beyond the top.
    * An invalid percent-encoding is encountered.
    """
    # Special case: empty path is converted to '/'
    if uri_path == "" or uri_path == "/":
        return "/"

    # All other paths must be absolute.
    if not uri_path.startswith("/"):
        raise ValueError("URI path is not absolute.")

    # Replace double slashes; this makes it easier to handle slashes at the
    # end.
    uri_path = _multislash.sub("/", uri_path)
    
    # Examine each path component for relative directories.
    components = uri_path.split("/")[1:]
    i = 0
    while i < len(components):
        # Fix % encodings.
        component = normalize_uri_path_component(components[i])
        components[i] = component
        
        if components[i] == ".":
            # Relative current directory.  Remove this.
            del components[i]

            # Don't increment i; with the deletion, we're now pointing to
            # the next element in the path.
        elif components[i] == "..":
            # Relative path: parent directory.  Remove this and the previous
            # component.
            if i == 0:
                # Not allowed at the beginning!
                raise ValueError("URI path attempts to go beyond root")
            del components[i-1:i+1]

            # Since we've deleted two components, we need to back up one to
            # examine what's now the next component.
            i -= 1
        else:
            # Leave it alone; proceed to the next component.
            i += 1
    
    return "/" + "/".join(components)