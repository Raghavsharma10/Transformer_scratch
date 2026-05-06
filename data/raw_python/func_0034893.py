def _replace_config(s, env):
    """[sectionname]optionname"""
    m = re.match(r"\[(.*?)\](.*)", s)
    if m:
        section, option = m.groups()
        expanded = env.config.get(section, option)
        return '\n'.join([expand_factor_conditions(e, env)
                          for e in expanded.split("\n")])
    else:
        raise ValueError()