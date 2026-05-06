def t_NUMBER(self, t):
        r'([0-9]+\.?[0-9]*|\.[0-9]+)([eE](\+|-)?[0-9]+)?'
        try:
            sv = t.value
            v = float(sv)
            iv = int(v)
            t.value = (iv if iv == v else v, sv)
        except ValueError:
            print("Number %s is too large!" % t.value)
            t.value = 0
        return t