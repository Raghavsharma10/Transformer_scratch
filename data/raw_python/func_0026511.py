def insert_nginx_service(definition):  # pragma: no cover
    """Insert a new nginx service definition"""

    config_file = '/etc/nginx/sites-available/hfos.conf'
    splitter = "### SERVICE DEFINITIONS ###"

    with open(config_file, 'r') as f:
        old_config = "".join(f.readlines())

    pprint(old_config)

    if definition in old_config:
        print("Service definition already inserted")
        return

    parts = old_config.split(splitter)
    print(len(parts))
    if len(parts) != 3:
        print("Nginx configuration seems to be changed and cannot be "
              "extended automatically anymore!")
        pprint(parts)
        return

    try:
        with open(config_file, "w") as f:
            f.write(parts[0])
            f.write(splitter + "\n")
            f.write(parts[1])
            for line in definition:
                f.write(line)
            f.write("\n    " + splitter)
            f.write(parts[2])
    except Exception as e:
        print("Error during Nginx configuration extension:", type(e), e)