def get_error(self, xml):
        '''Obtem do XML de resposta, o código e a descrição do erro.

        O XML corresponde ao corpo da resposta HTTP de código 500.

        :param xml: XML contido na resposta da requisição HTTP.

        :return: Tupla com o código e a descrição do erro contido no XML:
            (< codigo_erro>, < descricao_erro>)
        '''
        map = loads(xml)
        network_map = map['networkapi']
        error_map = network_map['erro']
        return int(error_map['codigo']), str(error_map['descricao'])