def create_key(self):
        """Create an API key."""
        print("Creating key. Please input the following options:")
        name = input("Key name (optional): ")
        print("To make this key more secure, you should restrict the IP addresses that can use it. ")
        print("To use with all IPs, leave blank or use 0.0.0.0/0.")
        print("To use with a single IP, append '/32', such as 207.39.29.22/32. ")
        print("See this reference on CIDR blocks: http://software77.net/cidr-101.html")
        cidr = input("CIDR (optional): ")
        key = self._curl_bitmex("/apiKey",
                                postdict={"name": name, "cidr": cidr, "enabled": True})

        print("Key created. Details:\n")
        print("API Key:    " + key["id"])
        print("Secret:     " + key["secret"])
        print("\nSafeguard your secret key! If somebody gets a hold of your API key and secret,")
        print("your account can be taken over completely.")
        print("\nKey generation complete.")