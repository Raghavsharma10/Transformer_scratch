def loads(xml, force_list=None):
    """Cria um dicionário com os dados do XML.

    O dicionário terá como chave o nome do nó root e como valor o conteúdo do nó root.
    Quando o conteúdo de um nó é uma lista de nós então o valor do nó será
    um dicionário com uma chave para cada nó.
    Entretanto, se existir nós, de um mesmo pai, com o mesmo nome, então eles serão
    armazenados em uma mesma chave do dicionário que terá como valor uma lista.

    O force_list deverá ter nomes de nós do XML que necessariamente terão seus
    valores armazenados em uma lista no dicionário de retorno.


    ::
        Por exemplo:
        xml_1 = &lt;?xml version="1.0" encoding="UTF-8"?&gt;
        &lt;networkapi versao="1.0"&gt;
        &lt;testes&gt;
        &lt;teste&gt;1&lt;teste&gt;
        &lt;teste&gt;2&lt;teste&gt;
        &lt;/testes&gt;
        &lt;/networkapi&gt;

        A chamada loads(xml_1), irá gerar o dicionário: {'networkapi':{'testes':{'teste':[1,2]}}}

        xml_2 = &lt;?xml version="1.0" encoding="UTF-8"?&gt;
        &lt;networkapi versao="1.0"&gt;
        &lt;testes&gt;
        &lt;teste&gt;1&lt;teste&gt;
        &lt;/testes&gt;
        &lt;/networkapi&gt;

        A chamada loads(xml_2), irá gerar o dicionário: {'networkapi':{'testes':{'teste':1}}}

        A chamada loads(xml_2, ['teste']), irá gerar o dicionário: {'networkapi':{'testes':{'teste':[1]}}}

        Ou seja, o XML_2 tem apenas um nó 'teste', porém, ao informar o parâmetro 'force_list'
        com o valor ['teste'], a chave 'teste', no dicionário, terá o valor dentro de uma lista.

    :param xml: XML
    :param force_list: Lista com os nomes dos nós do XML que deverão ter seus valores
        armazenados em lista dentro da chave do dicionário de retorno.

    :return: Dicionário com os nós do XML.

    :raise XMLErrorUtils: Representa um erro ocorrido durante o marshall ou unmarshall do XML.
    """
    if force_list is None:
        force_list = []

    try:
        xml = remove_illegal_characters(xml)
        doc = parseString(xml)
    except Exception as e:
        raise XMLErrorUtils(e, u'Falha ao realizar o parse do xml.')

    root = doc.documentElement

    map = dict()
    attrs_map = dict()

    if root.hasAttributes():
        attributes = root.attributes
        for i in range(attributes.length):
            attr = attributes.item(i)
            attrs_map[attr.nodeName] = attr.nodeValue

    map[root.nodeName] = _create_childs_map(root, force_list)

    return map