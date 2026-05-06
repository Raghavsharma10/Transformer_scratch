def _send(self):
        """ Send the webhook method """

        payload = self.payload
        sending_metadata = {'success': False}
        post_attributes = {'timeout': self.timeout}

        if self.custom_headers:
            post_attributes['headers'] = self.custom_headers
        if not post_attributes.get('headers', None):
            post_attributes['headers'] = {}
        post_attributes['headers']['Content-Type'] = self.encoding

        post_attributes['data'] = self.format_payload()

        if self.signing_secret:
            post_attributes['headers']['x-hub-signature'] = self.create_signature(post_attributes['data'], \
                                                                                  self.signing_secret)

        for i, wait in enumerate(range(len(self.attempts) - 1)):

            self.attempt = i + 1
            sending_metadata['attempt'] = self.attempt

            try:
                print(self.url)
                self.response = requests.post(self.url, **post_attributes)

                if sys.version > '3':
                    # Converts bytes object to str object in Python 3+
                    self.response_content = self.response.content.decode('utf-8')
                else:
                    self.response_content = self.response.content

                sending_metadata['status_code'] = self.response.status_code

                # anything with a 200 status code  is a success
                if self.response.status_code >= 200 and self.response.status_code < 300:
                    # Exit the sender method.  Here we provide the payload as a result.
                    #   This is useful for reporting.
                    self.notify("Attempt {}: Successfully sent webhook {}".format(
                        self.attempt, self.hash_value)
                    )
                    sending_metadata['response'] = self.response_content
                    sending_metadata['success'] = True
                    break
                else:
                    self.error = "Status code (%d). Message: %s" % (self.response.status_code, self.response.text)


            except Exception as ex:
                err_formatted = str(ex).replace('"',"'")
                sending_metadata['response'] = '{"status_code": 500, "status":"failure","error":"'+err_formatted+'"}'
                self.error = err_formatted

            self.notify("Attempt {}: Could not send webhook {}".format(
                    self.attempt, self.hash_value)
            )
            self.notify_debug("Webhook {}. Body: {}".format(
                    self.hash_value, self.payload)
            )

            # If last attempt
            if self.attempt == (len(self.attempts) - 1):
                self.notify_error("Failed to send webhook {}. Body: {}".format(
                    self.hash_value, self.payload)
                )
            else:
                # Wait a bit before the next attempt
                sleep(wait)

        sending_metadata['error'] = None if sending_metadata['success'] or not self.error else self.error
        sending_metadata['post_attributes'] = post_attributes
        merged_dict = sending_metadata.copy()
        if isinstance(payload, string_types):
            payload = {'payload': payload}

        # Add the hash value if there is one.
        if self.hash_value is not None and len(self.hash_value) > 0:
            payload['hash'] = self.hash_value

        merged_dict.update(payload)
        return merged_dict