def add_vip(
            self,
            id,
            real_name_sufixo,
            id_vlan,
            descricao_vlan,
            id_vlan_real,
            descricao_vlan_real,
            balanceadores,
            id_healthcheck_expect,
            finalidade,
            cliente,
            ambiente,
            cache,
            metodo_bal,
            persistencia,
            healthcheck_type,
            healthcheck,
            timeout,
            host,
            maxcon,
            dsr,
            bal_ativo,
            transbordos,
            portas,
            real_maps,
            id_requisicao_vip,
            areanegocio='Orquestra',
            nome_servico='Orquestra',
            l7_filter=None,
            reals_prioritys=None,
            reals_weights=None):
        """Adiciona um VIP na lista de VIPs para operação de inserir/alterar um grupo virtual.

        Os parâmetros abaixo somente são necessários para a operação de alteração:

            - 'real_maps': Deverá conter os reals atualmente criados para a requisição de VIP.
            - 'id_requisicao_vip': O identificador da requisição que deverá ser alterada.

        Os parâmetros abaixo somente são necessários para a operação de inserção:

            - 'id_vlan': Identificador da VLAN para criar o IP do VIP.
            - 'descricao_vlan': Descrição do IP do VIP.
            - balanceadores: Lista com os identificadores dos balanceadores que serão associados ao IP do VIP.

        :param id: Identificador do VIP utilizado pelo sistema de orquestração.
        :param real_name_sufixo: Sufixo utilizado para criar os reals_names dos equipamentos na requisição de VIP.
        :param id_vlan: Identificador da VLAN para criar um IP para o VIP.
        :param descricao_vlan: Descrição do IP que será criado para o VIP.
        :param id_vlan_real: Identificador da VLAN para criar os IPs dos equipamentos no VIP.
        :param descricao_vlan_real: Descrição dos IPs que serão criados para os equipamentos no VIP.
        :param balanceadores: Lista com os identificadores dos balanceadores que serão associados ao IP do VIP.
        :param id_healthcheck_expect: Identificador do healthcheck_expect para criar a requisição de VIP.
        :param finalidade: Finalidade da requisição de VIP.
        :param cliente: Cliente da requisição de VIP.
        :param ambiente: Ambiente da requisição de VIP.
        :param cache: Cache da requisição de VIP.
        :param metodo_bal: Método de balanceamento da requisição de VIP.
        :param persistencia: Persistência da requisição de VIP.
        :param healthcheck_type: Healthcheck_type da requisição de VIP.
        :param healthcheck: Healthcheck da requisição de VIP.
        :param timeout: Timeout da requisição de VIP.
        :param host: Host da requisição de VIP.
        :param maxcon: Máximo número de conexão da requisição de VIP.
        :param dsr: DSR da requisição de VIP.
        :param bal_ativo: Balanceador ativo da requisição de VIP.
        :param transbordos: Lista com os IPs dos transbordos da requisição de VIP.
        :param portas: Lista com as portas da requisição de VIP.
        :param real_maps: Lista dos mapas com os dados dos reals da requisição de VIP.
            Cada mapa deverá ter a estrutura: {'real_name':< real_name>, 'real_ip':< real_ip>}
        :param id_requisicao_vip: Identificador da requisição de VIP para operação de alterar um
            grupo virtual.
        :param areanegocio: Área de negócio para a requisição de VIP (é utilizado 'Orquestra' caso seja None).
        :param nome_servico: Nome do serviço para a requisição de VIP (é utilizado 'Orquestra' caso seja None).
        :param l7_filter: Filtro L7 para a requisição de VIP.
        :param reals_prioritys: Lista dos dados de prioridade dos reals da requisição de VIP (lista de zeros, caso seja None).
        :param reals_weights: Lista dos dados de pesos dos reals da requisição de VIP (lista de zeros, caso seja None).

        :return: None
        """
        vip_map = dict()

        vip_map['id'] = id
        # Causa erro na hora de validar os nomes de equipamentos (real servers)
        #vip_map['real_name_sufixo'] = real_name_sufixo
        vip_map['ip_real'] = {
            'id_vlan': id_vlan_real,
            'descricao': descricao_vlan_real}
        vip_map['ip'] = {'id_vlan': id_vlan, 'descricao': descricao_vlan}
        vip_map['balanceadores'] = {'id_equipamento': balanceadores}
        vip_map['id_healthcheck_expect'] = id_healthcheck_expect
        vip_map['finalidade'] = finalidade
        vip_map['cliente'] = cliente
        vip_map['ambiente'] = ambiente
        vip_map['cache'] = cache
        vip_map['metodo_bal'] = metodo_bal
        vip_map['persistencia'] = persistencia
        vip_map['healthcheck_type'] = healthcheck_type
        vip_map['healthcheck'] = healthcheck
        vip_map['timeout'] = timeout
        vip_map['host'] = host
        vip_map['maxcon'] = maxcon
        vip_map['dsr'] = dsr
        # Nao sao mais utilizados (bal_ativo e transbordos)
        #vip_map['bal_ativo'] = bal_ativo
        #vip_map['transbordos'] = {'transbordo': transbordos}
        vip_map['portas_servicos'] = {'porta': portas}
        vip_map['reals'] = {'real': real_maps}
        vip_map['areanegocio'] = areanegocio
        vip_map['nome_servico'] = nome_servico
        vip_map['l7_filter'] = l7_filter

        if reals_prioritys is not None:
            vip_map['reals_prioritys'] = {'reals_priority': reals_prioritys}
        else:
            vip_map['reals_prioritys'] = None

        if metodo_bal.upper() == 'WEIGHTED':
            if reals_weights is not None:
                vip_map['reals_weights'] = {'reals_weight': reals_weights}
            else:
                vip_map['reals_weights'] = None

        if id_requisicao_vip is not None:
            vip_map['requisicao_vip'] = {'id': id_requisicao_vip}

        self.lista_vip.append(vip_map)