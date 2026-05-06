def main():
    """For testing purpose"""
    tcp_adapter = TcpAdapter("192.168.1.3", name="HASS", activate_source=False)
    hdmi_network = HDMINetwork(tcp_adapter)
    hdmi_network.start()
    while True:
        for d in hdmi_network.devices:
            _LOGGER.info("Device: %s", d)

        time.sleep(7)