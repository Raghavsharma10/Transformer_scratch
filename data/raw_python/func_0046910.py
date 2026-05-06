def disable_key(self):
        """Disable an existing API Key."""
        print("This command will disable a enabled key.")
        apiKeyID = input("API Key ID: ")
        try:
            key = self._curl_bitmex("/apiKey/disable",
                                    postdict={"apiKeyID": apiKeyID})
            print("Key with ID %s disabled." % key["id"])
        except:
            print("Unable to disable key, please try again.")
            self.disable_key()