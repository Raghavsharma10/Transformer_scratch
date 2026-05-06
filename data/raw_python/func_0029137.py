def _get_fillcolor(resource_type, properties, known_sg=[], open_sg=[]):
    """Determine fillcolor for resources (public ones in this case)
    """
    fillcolor = None
    # check security groups
    if 'SecurityGroups' in properties:
        # check for external security groups
        for sg in properties['SecurityGroups']:
            if 'Ref' in sg and (sg['Ref'] not in known_sg):
                fillcolor = 'yellow'
                break

        # check for open security groups
        for osg in open_sg:
            if {'Ref': osg} in properties['SecurityGroups']:
                fillcolor = 'red'
                break

    # LoadBalancer
    if resource_type == 'LoadBalancer':
        if ('Scheme' not in properties) or \
                        properties['Scheme'] == 'internet-facing':
            fillcolor = 'red'

    return fillcolor