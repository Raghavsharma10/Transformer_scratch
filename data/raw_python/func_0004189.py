def add_vip_incremento(self, id):
        """Adiciona um vip à especificação do grupo virtual.

        :param id: Identificador de referencia do VIP.
        """
        vip_map = dict()

        vip_map['id'] = id

        self.lista_vip.append(vip_map)