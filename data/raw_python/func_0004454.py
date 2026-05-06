def inserir(self, nome):
        """Insere um novo grupo de equipamento e retorna o seu identificador.

        :param nome: Nome do grupo de equipamento.

        :return: Dicionário com a seguinte estrutura: {'grupo': {'id': < id >}}

        :raise InvalidParameterError: Nome do grupo é nulo ou vazio.
        :raise NomeGrupoEquipamentoDuplicadoError: Nome do grupo de equipmaneto duplicado.
        :raise DataBaseError: Falha na networkapi ao acessar o banco de dados.
        :raise XMLError: Falha na networkapi ao ler o XML de requisição ou gerar o XML de resposta.
        """
        egrupo_map = dict()
        egrupo_map['nome'] = nome

        code, xml = self.submit({'grupo': egrupo_map}, 'POST', 'egrupo/')

        return self.response(code, xml)