def send_challenge_response(self, response_plain):
        """Send a challenge response to server"""

        # Get a basic stanza body
        body = self.get_body()

        # Create a response tag and add the response content on it
        #   using base64 encoding
        response_node = ET.SubElement(body, 'response')
        response_node.set('xmlns', XMPP_SASL_NS)
        response_node.text = base64.b64encode(response_plain)

        # Send the challenge response to server
        resp_root = ET.fromstring(self.send_request(body))
        return resp_root