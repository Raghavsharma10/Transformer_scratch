def windows_process_priority_format(instance):
    """Ensure the 'priority' property of windows-process-ext ends in '_CLASS'.
    """
    class_suffix_re = re.compile(r'.+_CLASS$')
    for key, obj in instance['objects'].items():
        if 'type' in obj and obj['type'] == 'process':
            try:
                priority = obj['extensions']['windows-process-ext']['priority']
            except KeyError:
                continue
            if not class_suffix_re.match(priority):
                yield JSONError("The 'priority' property of object '%s' should"
                                " end in '_CLASS'." % key, instance['id'],
                                'windows-process-priority-format')