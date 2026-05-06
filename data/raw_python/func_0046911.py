def delete_key(self):
        """Delete an existing API Key."""
        print("This command will delete an API key.")
        apiKeyID = input("API Key ID: ")
        try:
            self._curl_bitmex("/apiKey/",
                              postdict={"apiKeyID": apiKeyID}, verb='DELETE')
            print("Key with ID %s disabled." % apiKeyID)
        except:
            print("Unable to delete key, please try again.")
            self.delete_key()