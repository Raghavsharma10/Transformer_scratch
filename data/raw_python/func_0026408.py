def install_service(instance, dbhost, dbname, port):
    """Install systemd service configuration"""

    _check_root()

    log("Installing systemd service")

    launcher = os.path.realpath(__file__).replace('manage', 'launcher')
    executable = sys.executable + " " + launcher
    executable += " --instance " + instance
    executable += " --dbname " + dbname + " --dbhost " + dbhost
    executable += " --port " + port
    executable += " --dolog --logfile /var/log/hfos-" + instance + ".log"
    executable += " --logfileverbosity 30 -q"

    definitions = {
        'instance': instance,
        'executable': executable
    }
    service_name = 'hfos-' + instance + '.service'

    write_template_file(os.path.join('dev/templates', service_template),
                        os.path.join('/etc/systemd/system/', service_name),
                        definitions)

    Popen([
        'systemctl',
        'enable',
        service_name
    ])

    log('Launching service')

    Popen([
        'systemctl',
        'start',
        service_name
    ])

    log("Done: Install Service")