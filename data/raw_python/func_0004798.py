def get_similar(self, example, max_similars=3, similarity_cutoff=None,
                    term_diff_max_rank=10, filter_list=None,
                    term_diff_cutoff=None):
        """Devuelve textos similares al ejemplo dentro de los textos entrenados.

        Nota:
            Usa la distancia de coseno del vector de features TF-IDF

        Args:
            example (str): Se espera un id de texto o un texto a partir del
                cual se buscaran otros textos similares.
            max_similars (int, optional): Cantidad de textos similares a
                devolver.
            similarity_cutoff (float, optional): Valor umbral de similaridad
                para definir que dos textos son similares entre si.
            term_diff_max_rank (int, optional): Este valor sirve para controlar
                el umbral con el que los terminos son considerados importantes
                a la hora de recuperar textos (no afecta el funcionamiento de
                que textos se consideran cercanos, solo la cantidad de terminos
                que se devuelven en best_words).
            filter_list (list): Lista de ids de textos en la cual buscar textos
                similares.
            term_diff_cutoff (float): Deprecado. Se quitara en el futuro.

        Returns:
            tuple (list, list, list): (text_ids, sorted_dist, best_words)
                text_ids (list of str): Devuelve los ids de los textos
                    sugeridos.
                sorted_dist (list of float): Devuelve la distancia entre las
                    opciones sugeridas y el ejemplo dado como entrada.
                best_words (list of list): Para cada sugerencia devuelve las
                    palabras mas relevantes que se usaron para seleccionar esa
                    sugerencia.
        """

        if term_diff_cutoff:
            warnings.warn('Deprecado. Quedo sin uso. Se quitara en el futuro.',
                          DeprecationWarning)
        if filter_list:
            if max_similars > len(filter_list):
                raise ValueError("No se pueden pedir mas sugerencias que la \
                                  cantidad de textos en `filter_list`.")
            else:
                filt_idx = np.in1d(self.ids, filter_list)

        elif max_similars > self.term_mat.shape[0]:
            raise ValueError("No se pueden pedir mas sugerencias que la \
                              cantidad de textos que hay almacenados.")
        else:
            filt_idx = np.ones(len(self.ids), dtype=bool)
        # Saco los textos compuestos solo por stop_words
        good_ids = np.array(np.sum(self.term_mat, 1) > 0).squeeze()
        filt_idx = filt_idx & good_ids
        filt_idx_to_general_idx = np.flatnonzero(filt_idx)
        if example in self.ids:
            index = self.ids == example
            exmpl_vec = self.tfidf_mat[index, :]
            distances = np.squeeze(pairwise_distances(self.tfidf_mat[filt_idx],
                                                      exmpl_vec))
            # Pongo la distancia a si mismo como inf, par que no se devuelva a
            # si mismo como una opcion
            if filter_list and example in filter_list:
                distances[filter_list.index(example)] = np.inf
            elif not filter_list:
                idx_example = np.searchsorted(self.ids, example)
                filt_idx_example = np.searchsorted(np.flatnonzero(filt_idx),
                                                   idx_example)
                distances[filt_idx_example] = np.inf
        else:
            exmpl_vec = self.vectorizer.transform([example])  # contar terminos
            exmpl_vec = self.transformer.transform(exmpl_vec)  # calcular tfidf
            distances = np.squeeze(pairwise_distances(self.tfidf_mat[filt_idx],
                                                      exmpl_vec))
        if np.sum(exmpl_vec) == 0:
            return [], [], []
        sorted_indices = np.argsort(distances)
        closest_n = sorted_indices[:max_similars]
        sorted_dist = distances[closest_n]
        if similarity_cutoff:
            closest_n = closest_n[sorted_dist < similarity_cutoff]
            sorted_dist = sorted_dist[sorted_dist < similarity_cutoff]
        best_words = []

        # Calculo palabras relevantes para cada sugerencia
        best_example = np.squeeze(exmpl_vec.toarray())
        sorted_example_weights = np.flipud(np.argsort(best_example))
        truncated_max_rank = min(term_diff_max_rank, np.sum(best_example > 0))
        best_example_words = sorted_example_weights[:truncated_max_rank]
        for suggested in closest_n:
            suggested_idx = filt_idx_to_general_idx[suggested]
            test_vec = np.squeeze(self.tfidf_mat[suggested_idx, :].toarray())
            sorted_test_weights = np.flipud(np.argsort(test_vec))
            truncated_max_rank = min(term_diff_max_rank,
                                     np.sum(test_vec > 0))
            best_test = sorted_test_weights[:truncated_max_rank]
            best_words_ids = np.intersect1d(best_example_words, best_test)
            best_words.append([k for k, v in
                               self.vectorizer.vocabulary_.items()
                               if v in best_words_ids])

        # Filtro dentro de las buscadas
        if filter_list:
            text_ids = self.ids[filt_idx_to_general_idx[closest_n]]
        else:
            text_ids = self.ids[closest_n]
        return list(text_ids), list(sorted_dist), best_words