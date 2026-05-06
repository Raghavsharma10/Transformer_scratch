def salvar(self, destino=None, prefix='tmp', suffix='-sat.log'):
        """Salva o arquivo de log decodificado.

        :param str destino: (Opcional) Caminho completo para o arquivo onde os
            dados dos logs deverão ser salvos. Se não informado, será criado
            um arquivo temporário via :func:`tempfile.mkstemp`.

        :param str prefix: (Opcional) Prefixo para o nome do arquivo. Se não
            informado será usado ``"tmp"``.

        :param str suffix: (Opcional) Sufixo para o nome do arquivo. Se não
            informado será usado ``"-sat.log"``.

        :return: Retorna o caminho completo para o arquivo salvo.
        :rtype: str

        :raises IOError: Se o destino for informado e o arquivo já existir.
        """
        if destino:
            if os.path.exists(destino):
                raise IOError((errno.EEXIST, 'File exists', destino,))
            destino = os.path.abspath(destino)
            fd = os.open(destino, os.O_EXCL|os.O_CREAT|os.O_WRONLY)
        else:
            fd, destino = tempfile.mkstemp(prefix=prefix, suffix=suffix)

        os.write(fd, self.conteudo())
        os.fsync(fd)
        os.close(fd)

        return os.path.abspath(destino)