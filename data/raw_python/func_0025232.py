def status(self):
        """Nome amigável do campo ``ESTADO_OPERACAO``, conforme a "Tabela de
        Informações do Status do SAT".
        """
        for valor, rotulo in ESTADOS_OPERACAO:
            if self.ESTADO_OPERACAO == valor:
                return rotulo
        return u'(desconhecido: {})'.format(self.ESTADO_OPERACAO)