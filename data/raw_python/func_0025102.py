def _carregar(self):
        """Carrega (ou recarrega) a biblioteca SAT. Se a convenção de chamada
        ainda não tiver sido definida, será determinada pela extensão do
        arquivo da biblioteca.

        :raises ValueError: Se a convenção de chamada não puder ser determinada
            ou se não for um valor válido.
        """
        if self._convencao is None:
            if self._caminho.endswith(('.DLL', '.dll')):
                self._convencao = constantes.WINDOWS_STDCALL
            else:
                self._convencao = constantes.STANDARD_C

        if self._convencao == constantes.STANDARD_C:
            loader = ctypes.CDLL

        elif self._convencao == constantes.WINDOWS_STDCALL:
            loader = ctypes.WinDLL

        else:
            raise ValueError('Convencao de chamada desconhecida: {!r}'.format(
                    self._convencao))

        self._libsat = loader(self._caminho)