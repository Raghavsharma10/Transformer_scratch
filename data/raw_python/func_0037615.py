def _process_response_xml(self, response_xml):
        '''
        Processa o xml de resposta e caso não existam erros retorna um
        dicionario com o codigo e data.

        :return: dictionary
        '''
        result = {}
        xml = ElementTree.fromstring(response_xml)
        if xml.tag == 'errors':
            logger.error(
                u'Erro no pedido de pagamento ao PagSeguro.' +
                ' O xml de resposta foi: %s' % response_xml)
            errors_message = u'Ocorreu algum problema com os dados do pagamento: '
            for error in xml.findall('error'):
                error_code = error.find('code').text
                error_message = error.find('message').text
                errors_message += u'\n (code=%s) %s' % (error_code,
                                                        error_message)
            raise PagSeguroPaymentException(errors_message)

        if xml.tag == 'checkout':
            result['code'] = xml.find('code').text

            try:
                xml_date = xml.find('date').text
                result['date'] = dateutil.parser.parse(xml_date)
            except:
                logger.exception(u'O campo date não foi encontrado ou é invalido')
                result['date'] = None
        else:
            raise PagSeguroPaymentException(
                u'Erro ao processar resposta do pagamento: tag "checkout" nao encontrada no xml de resposta')
        return result