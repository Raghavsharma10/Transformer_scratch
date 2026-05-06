def select_config_sections(configfile_sections, desired_section_patterns):
    """Select a subset of the sections in a configuration file by using
    a list of section names of list of section name patters
    (supporting :mod:`fnmatch` wildcards).

    :param configfile_sections: List of config section names (as strings).
    :param desired_section_patterns:
    :return: List of selected section names or empty list (as generator).
    """
    for section_name in configfile_sections:
        for desired_section_pattern in desired_section_patterns:
            if fnmatch(section_name, desired_section_pattern):
                yield section_name