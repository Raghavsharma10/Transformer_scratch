def determine_inside_container(self):
        """
            Set self.in_container if we're inside a container
             * Inside container
             * Current token starts a new container
             * Current token ends all containers
        """
        tokenum, value = self.current.tokenum, self.current.value
        ending_container = False
        starting_container = False

        if tokenum == OP:
            # Record when we're inside a container of some sort (tuple, list, dictionary)
            # So that we can care about that when determining what to do with whitespace
            if value in ['(', '[', '{']:
                # add to the stack because we started a list
                self.containers.append(value)
                starting_container = True

            elif value in [')', ']', '}']:
                # not necessary to check for correctness
                self.containers.pop()
                ending_container = True

        self.just_ended_container = not len(self.containers) and ending_container
        self.just_started_container = len(self.containers) == 1 and starting_container
        self.in_container = len(self.containers) or self.just_ended_container or self.just_started_container