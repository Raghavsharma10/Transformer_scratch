def _make_text_vectors(self, examples):
        """Funcion para generar los vectores tf-idf de una lista de textos.

        Args:
            examples (list or str): Se espera un ejemplo o una lista de:
                o bien ids, o bien textos.
        Returns:
            textvec (sparse matrix): Devuelve una matriz sparse que contiene
                los vectores TF-IDF para los ejemplos que se pasan de entrada.
                El tamaño de la matriz es de (N, T) donde N es la cantidad de
                ejemplos y T es la cantidad de términos en el vocabulario.
        """
        if isinstance(examples, str):
            if examples in self.ids:
                textvec = self.tfidf_mat[self.ids == examples, :]
            else:
                textvec = self.vectorizer.transform([examples])
                textvec = self.transformer.transform(textvec)
        elif type(examples) is list:
            if all(np.in1d(examples, self.ids)):
                textvec = self.tfidf_mat[np.in1d(self.ids, examples)]
            elif not any(np.in1d(examples, self.ids)):
                textvec = self.vectorizer.transform(examples)
                textvec = self.transformer.transform(textvec)
            else:
                raise ValueError("Las listas de ejemplos deben ser todos ids\
                                  de textos almacenados o todos textos planos")
        else:
            raise TypeError("Los ejemplos no son del tipo de dato adecuado.")

        return textvec