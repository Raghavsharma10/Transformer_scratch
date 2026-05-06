def control_iter(base_dir, control_name=CONTROL_NAME):
    """
    Generate the names of all control files under base_dir
    """
    controls = (os.path.join(p, control_name) for p, _, fs in os.walk(base_dir)
                if control_name in fs)
    return controls