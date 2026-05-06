def populate(self, 
        size, 
        names_library=None, 
        reuse_names=False,
        random_branches=False, 
        branch_range=(0, 1),
        support_range=(0, 1)):
        """
        Generates a random topology by populating current node.

        :argument None names_library: If provided, names library
          (list, set, dict, etc.) will be used to name nodes.

        :argument False reuse_names: If True, node names will not be
          necessarily unique, which makes the process a bit more
          efficient.

        :argument False random_branches: If True, branch distances and support
          values will be randomized.

        :argument (0,1) branch_range: If random_branches is True, this
        range of values will be used to generate random distances.

        :argument (0,1) support_range: If random_branches is True,
        this range of values will be used to generate random branch
        support values.

        """
        NewNode = self.__class__

        if len(self.children) > 1:
            connector = NewNode()
            for ch in self.get_children():
                ch.detach()
                connector.add_child(child = ch)
            root = NewNode()
            self.add_child(child = connector)
            self.add_child(child = root)
        else:
            root = self

        next_deq = deque([root])
        for i in range(size-1):
            if random.randint(0, 1):
                p = next_deq.pop()
            else:
                p = next_deq.popleft()

            c1 = p.add_child()
            c2 = p.add_child()
            next_deq.extend([c1, c2])
            if random_branches:
                c1.dist = random.uniform(*branch_range)
                c2.dist = random.uniform(*branch_range)
                c1.support = random.uniform(*branch_range)
                c2.support = random.uniform(*branch_range)
            else:
                c1.dist = 1.0
                c2.dist = 1.0
                c1.support = 1.0
                c2.support = 1.0

        # next contains leaf nodes
        charset = "abcdefghijklmnopqrstuvwxyz"
        if names_library:
            names_library = deque(names_library)
        else:
            avail_names = itertools.combinations_with_replacement(charset, 10)
        for n in next_deq:
            if names_library:
                if reuse_names:
                    tname = random.sample(names_library, 1)[0]
                else:
                    tname = names_library.pop()
            else:
                tname = ''.join(next(avail_names))
            n.name = tname