def network_traffic_ports(instance):
    """Ensure network-traffic objects contain both src_port and dst_port.
    """
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] == 'network-traffic' and
                ('src_port' not in obj or 'dst_port' not in obj)):
            yield JSONError("The Network Traffic object '%s' should contain "
                            "both the 'src_port' and 'dst_port' properties."
                            % key, instance['id'], 'network-traffic-ports')