def jd(api_response):
        """
        JD (JSON Dump) function. Meant for quick pretty-printing of CloudGenix Response objects.

        Example: `jd(cgx_sess.get.sites())`

        **Returns:** No Return, directly prints all output.
        """
        try:
            # attempt to print the cgx_content. should always be a Dict if it exists.
            print(json.dumps(api_response.cgx_content, indent=4))
        except (TypeError, ValueError, AttributeError):
            # cgx_content did not exist, or was not JSON serializable. Try pretty printing the base obj.
            try:
                print(json.dumps(api_response, indent=4))
            except (TypeError, ValueError, AttributeError):
                # Same issue, just raw print the passed data. Let any exceptions happen here.
                print(api_response)
        return