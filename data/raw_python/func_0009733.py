def send(self, messages):
        """Send a SMS message, or an array of SMS messages"""

        tmpSms = SMS(to='', message='')
        if str(type(messages)) == str(type(tmpSms)):
            messages = [messages]

        xml_root = self.__init_xml('Message')
        wrapper_id = 0

        for m in messages:
            m.wrapper_id = wrapper_id
            msg = self.__build_sms_data(m)
            sms = etree.SubElement(xml_root, 'SMS')
            for sms_element in msg:
                element = etree.SubElement(sms, sms_element)
                element.text = msg[sms_element]

        # print etree.tostring(xml_root)
        response = clockwork_http.request(SMS_URL, etree.tostring(xml_root, encoding='utf-8'))
        response_data = response['data']

        # print response_data
        data_etree = etree.fromstring(response_data)

        # Check for general error
        err_desc = data_etree.find('ErrDesc')
        if err_desc is not None:
            raise clockwork_exceptions.ApiException(err_desc.text, data_etree.find('ErrNo').text)

        # Return a consistent object
        results = []
        for sms in data_etree:
            matching_sms = next((s for s in messages if str(s.wrapper_id) == sms.find('WrapperID').text), None)
            new_result = SMSResponse(
                sms = matching_sms,
                id = '' if sms.find('MessageID') is None else sms.find('MessageID').text,
                error_code = 0 if sms.find('ErrNo') is None else sms.find('ErrNo').text,
                error_message = '' if sms.find('ErrDesc') is None else sms.find('ErrDesc').text,
                success = True if sms.find('ErrNo') is None else (sms.find('ErrNo').text == 0)
            )
            results.append(new_result)

        if len(results) > 1:
            return results

        return results[0]