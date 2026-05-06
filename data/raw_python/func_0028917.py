def _get_autoscaling_min_max(template, parameters, asg_name):
    """Helper to extract the configured MinSize, MaxSize attributes from the
    template.

    :param template: cloudformation template (json)
    :param parameters: list of {'ParameterKey': 'x1', 'ParameterValue': 'y1'}
    :param asg_name: logical resource name of the autoscaling group
    :return: MinSize, MaxSize
    """
    params = {e['ParameterKey']: e['ParameterValue'] for e in parameters}
    asg = template.get('Resources', {}).get(asg_name, None)
    if asg:
        assert asg['Type'] == 'AWS::AutoScaling::AutoScalingGroup'
        min = asg.get('Properties', {}).get('MinSize', None)
        max = asg.get('Properties', {}).get('MaxSize', None)
        if 'Ref' in min:
            min = params.get(min['Ref'], None)
        if 'Ref' in max:
            max = params.get(max['Ref'], None)
        if min and max:
            return int(min), int(max)