def add_equipamento_remove(self, id, id_ip, ids_ips_vips):
        '''Adiciona um equipamento na lista de equipamentos para operação de remover um grupo virtual.

        :param id: Identificador do equipamento.
        :param id_ip: Identificador do IP do equipamento.
        :param ids_ips_vips: Lista com os identificadores de IPs criados para cada VIP e associados ao
            equipamento.

        :return: None
        '''
        equipament_map = dict()
        equipament_map['id'] = id
        equipament_map['id_ip'] = id_ip
        equipament_map['vips'] = {'id_ip_vip': ids_ips_vips}

        self.lista_equipamentos_remove.append(equipament_map)