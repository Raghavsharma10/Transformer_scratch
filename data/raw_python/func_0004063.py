def get_version(extension, workflow_file):
    '''Determines the version of a .py, .wdl, or .cwl file.'''
    if extension == 'py' and two_seven_compatible(workflow_file):
        return '2.7'
    elif extension == 'cwl':
        return yaml.load(open(workflow_file))['cwlVersion']
    else:  # Must be a wdl file.
        # Borrowed from https://github.com/Sage-Bionetworks/synapse-orchestrator/blob/develop/synorchestrator/util.py#L142
        try:
            return [l.lstrip('version') for l in workflow_file.splitlines() if 'version' in l.split(' ')][0]
        except IndexError:
            return 'draft-2'