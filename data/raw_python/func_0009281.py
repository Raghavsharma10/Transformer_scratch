def normalize_string(mac_type, resource, content_hash):
    """Serializes mac_type and resource into a HAWK string."""

    normalized = [
        'hawk.' + str(HAWK_VER) + '.' + mac_type,
        normalize_header_attr(resource.timestamp),
        normalize_header_attr(resource.nonce),
        normalize_header_attr(resource.method or ''),
        normalize_header_attr(resource.name or ''),
        normalize_header_attr(resource.host),
        normalize_header_attr(resource.port),
        normalize_header_attr(content_hash or '')
    ]

    # The blank lines are important. They follow what the Node Hawk lib does.

    normalized.append(normalize_header_attr(resource.ext or ''))

    if resource.app:
        normalized.append(normalize_header_attr(resource.app))
        normalized.append(normalize_header_attr(resource.dlg or ''))

    # Add trailing new line.
    normalized.append('')

    normalized = '\n'.join(normalized)

    return normalized