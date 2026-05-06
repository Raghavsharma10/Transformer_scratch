def buttons(self, master):
        """Adds 'OK' and 'Cancel' buttons to standard button frame.

        Override if need for different configuration.
        """

        subframe = tk.Frame(master)
        subframe.pack(side=tk.RIGHT)

        ttk.Button(
            subframe,
            text="OK",
            width=10,
            command=self.ok,
            default=tk.ACTIVE
        ).pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(
            subframe,
            text="Cancel",
            width=10,
            command=self.cancel,
            default=tk.ACTIVE
        ).pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)