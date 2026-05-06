def dumps(map, root_name, root_attributes=None):
    """Cria um string no formato XML a partir dos elementos do map.

    Os elementos do mapa serão nós filhos do root_name.

    Cada chave do map será um Nó no XML. E o valor da chave será o conteúdo do Nó.

    Exemplos:

    ::

        - Mapa: {'networkapi':1}
          XML: &lt;?xml version="1.0" encoding="UTF-8"?&gt;&lt;networkapi&gt;1&lt;/networkapi&gt;

        - Mapa: {'networkapi':{'teste':1}}
          XML:  &lt;?xml version="1.0" encoding="UTF-8"?&gt;
          &lt;networkapi&gt;
          &lt;teste&gt;1&lt;/teste&gt;
          &lt;/networkapi&gt;

        - Mapa: {'networkapi':{'teste01':01, 'teste02':02}}
          XML: &lt;?xml version="1.0" encoding="UTF-8"?&gt;
          &lt;networkapi&gt;
          &lt;teste01&gt;01&lt;/teste01&gt;
          &lt;teste02&gt;02&lt;/teste02&gt;
          &lt;/networkapi&gt;

        - Mapa: {'networkapi':{'teste01':01, 'teste02':[02,03,04]}}
          XML: &lt;?xml version="1.0" encoding="UTF-8"?&gt;
          &lt;networkapi&gt;
          &lt;teste01&gt;01&lt;/teste01&gt;
          &lt;teste02&gt;02&lt;/teste02&gt;
          &lt;teste02&gt;03&lt;/teste02&gt;
          &lt;teste02&gt;04&lt;/teste02&gt;
          &lt;/networkapi&gt;

        - Mapa: {'networkapi':{'teste01':01, 'teste02':{'a':1, 'b':2}}}
          XML: &lt;?xml version="1.0" encoding="UTF-8"?&gt;
          &lt;networkapi&gt;
          &lt;teste01&gt;01&lt;/teste01&gt;
          &lt;teste02&gt;
          &lt;a&gt;1&lt;/a&gt;
          &lt;b&gt;2&lt;/b&gt;
          &lt;/teste02&gt;
          &lt;/networkapi&gt;

    :param map: Dicionário com os dados para serem convertidos em XML.
    :param root_name: Nome do nó root do XML.
    :param root_attributes: Dicionário com valores para serem adicionados como atributos
        para o nó root.

    :return: XML

    :raise XMLErrorUtils: Representa um erro ocorrido durante o marshall ou unmarshall do XML.
    :raise InvalidNodeNameXMLError: Nome inválido para representá-lo como uma TAG de XML.
    :raise InvalidNodeTypeXMLError: "Tipo inválido para o conteúdo de uma TAG de XML.
    """
    xml = ''
    try:
        implementation = getDOMImplementation()
    except ImportError as i:
        raise XMLErrorUtils(i, u'Erro ao obter o DOMImplementation')

    doc = implementation.createDocument(None, root_name, None)

    try:
        root = doc.documentElement

        if (root_attributes is not None):
            for key, value in root_attributes.iteritems():
                attribute = doc.createAttribute(key)
                attribute.nodeValue = value
                root.setAttributeNode(attribute)

        _add_nodes_to_parent(map, root, doc)

        xml = doc.toxml('UTF-8')
    except InvalidCharacterErr as i:
        raise InvalidNodeNameXMLError(
            i,
            u'Valor inválido para nome de uma TAG de XML: %s' %
            root_name)
    finally:
        doc.unlink()

    return xml