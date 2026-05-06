def verify_ws2p_head(self, head: Any) -> bool:
        """
        Check specified document
        :param Any head:
        :return:
        """
        signature = base64.b64decode(head.signature)
        inline = head.inline()
        prepended = signature + bytes(inline, 'ascii')

        try:
            self.verify(prepended)
            return True
        except ValueError:
            return False