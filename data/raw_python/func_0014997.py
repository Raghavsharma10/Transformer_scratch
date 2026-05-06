def _send(self, message):
        """ Given a message send it to the graphite server. """

        # An option to lowercase the entire message
        if self.lowercase_metric_names:
            message = message.lower()

        # convert the message into a pickled payload.
        message = self.str2listtuple(message)

        try:
            self.socket.sendall(message)
        # Capture missing socket.
        except socket.gaierror as error:
            raise GraphiteSendException(
                "Failed to send data to %s, with error: %s" %
                (self.addr, error))  # noqa

        # Capture socket closure before send.
        except socket.error as error:
            raise GraphiteSendException(
                "Socket closed before able to send data to %s, "
                "with error: %s" %
                (self.addr, error))  # noqa

        except Exception as error:
            raise GraphiteSendException(
                "Unknown error while trying to send data down socket to %s, "
                "error: %s" %
                (self.addr, error))  # noqa

        return "sent %d long pickled message" % len(message)