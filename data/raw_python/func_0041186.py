def chunks(lista, size):
    """Yield successive n-sized chunks from l."""
    if not isinstance(lista, list):
        logger.error('Erron in chunks, arg introduced is not a list.', lista=lista)
        return lista

    for i in range(0, len(lista), size):
        yield lista[i:i + size]