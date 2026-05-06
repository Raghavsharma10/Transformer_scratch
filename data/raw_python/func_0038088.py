def Email(msg=None):
    '''
    Valida endereços de email
    '''
    def f(v):
        if re.match("[\w\.\-]*@[\w\.\-]*\.\w+", str(v)):
            return str(v)
        else:
            raise Invalid(msg or ("Endereco de email invalido"))
    return f