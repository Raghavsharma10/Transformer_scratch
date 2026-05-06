def main(host, password, username):
    """Console script for tplink."""
    client = tplink.TpLinkClient(password)
    devices = client.get_connected_devices()
    click.echo(json.dumps(devices, indent=4))
    return 0