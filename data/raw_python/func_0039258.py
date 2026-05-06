def _is_private_ip(ip):
	"""
	Taken from https://stackoverflow.com/a/39656628/268006

	Check if the IP belongs to private network blocks.
	@param ip: IP address to verify.
	@return: boolean representing whether the IP belongs or not to
			 a private network block.
	"""
	networks = [
		"0.0.0.0/8",
		"10.0.0.0/8",
		"100.64.0.0/10",
		"127.0.0.0/8",
		"169.254.0.0/16",
		"172.16.0.0/12",
		"192.0.0.0/24",
		"192.0.2.0/24",
		"192.88.99.0/24",
		"192.168.0.0/16",
		"198.18.0.0/15",
		"198.51.100.0/24",
		"203.0.113.0/24",
		"240.0.0.0/4",
		"255.255.255.255/32",
		"224.0.0.0/4",
	]

	for network in networks:
		try:
			ipaddr = struct.unpack(">I", socket.inet_aton(ip))[0]

			netaddr, bits = network.split("/")

			network_low = struct.unpack(">I", socket.inet_aton(netaddr))[0]
			network_high = network_low | 1 << (32 - int(bits)) - 1

			if ipaddr <= network_high and ipaddr >= network_low:
				return True
		except Exception:
			continue

	return False