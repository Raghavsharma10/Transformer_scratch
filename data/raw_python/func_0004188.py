def add_vip_remove(self, id_ip, id_equipamentos):
        '''Adiciona um VIP na lista de VIPs para operação de remover um grupo virtual.

        :param id_ip: Identificador do IP criado para o VIP.
        :param id_equipamentos: Lista com os identificadores dos balanceadores associados ao IP do VIP.

        :return: None
        '''
        vip_map = dict()
        vip_map['id_ip'] = id_ip
        vip_map['balanceadores'] = {'id_equipamento': id_equipamentos}

        self.lista_vip_remove.append(vip_map)