def checar(cliente_sat):
    """
    Checa em sequência os alertas registrados (veja :func:`registrar`) contra os
    dados da consulta ao status operacional do equipamento SAT. Este método irá
    então resultar em uma lista dos alertas ativos.

    :param cliente_sat: Uma instância de
        :class:`satcfe.clientelocal.ClienteSATLocal` ou
        :class:`satcfe.clientesathub.ClienteSATHub` onde será invocado o método
        para consulta ao status operacional do equipamento SAT.

    :rtype: list
    """

    resposta = cliente_sat.consultar_status_operacional()
    alertas = []
    for classe_alerta in AlertaOperacao.alertas_registrados:
        alerta = classe_alerta(resposta)
        if alerta.checar():
            alertas.append(alerta)
    return alertas