def configurar_interface_de_rede(self, configuracao):
        """Sobrepõe :meth:`~satcfe.base.FuncoesSAT.configurar_interface_de_rede`.

        :return: Uma resposta SAT padrão.
        :rtype: satcfe.resposta.padrao.RespostaSAT
        """
        retorno = super(ClienteSATLocal, self).\
                configurar_interface_de_rede(configuracao)
        return RespostaSAT.configurar_interface_de_rede(retorno)