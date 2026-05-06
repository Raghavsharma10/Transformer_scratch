def analyze_reply_code(self, xml_response_dict):
        """
        Checks the RETS Response Code and handles non-zero answers.
        :param xml_response_dict:
        :return: None
        """
        if 'RETS-STATUS' in xml_response_dict:
            attributes = self.get_attributes(xml_response_dict['RETS-STATUS'])
            reply_code = attributes['ReplyCode']
            reply_text = attributes.get('ReplyText', 'RETS did not supply a Reply Text.')

            logger.debug("Received ReplyCode of {0!s} from the RETS Server: {0!s}".format(reply_code, reply_text))
            if reply_code != '0':
                raise RETSException(reply_text, reply_code)

        elif 'RETS' not in xml_response_dict:  # pragma: no cover
            raise RETSException("The <RETS> tag was expected in the response XML but it was not found.")

        attributes = self.get_attributes(input_dict=xml_response_dict['RETS'])
        if 'ReplyCode' not in attributes:  # pragma: no cover
            # The RETS server did not return a response code.
            return True

        reply_code = attributes['ReplyCode']
        reply_text = attributes.get('ReplyText', 'RETS did not supply a Reply Text.')

        logger.debug("Received ReplyCode of {0!s} from the RETS Server: {0!s}".format(reply_code, reply_text))
        if reply_code != '0':
            raise RETSException(reply_text, reply_code)