def _set_signatures(self):
        """Sets return and parameter types for the foreign C functions."""

        # We currently pass structs as void pointers.
        code_t = ctypes.c_int
        gpr_t = ctypes.c_int32
        int32_t = ctypes.c_int32
        node_p = ctypes.c_void_p
        pointer_t = ctypes.c_void_p
        state_p = ctypes.c_void_p
        void = None

        def sig(rettype, fname, *ptypes):
            func = getattr(self.lib, fname)
            func.restype = rettype
            func.argtypes = ptypes

        sig(node_p, "_jit_arg", state_p)
        sig(node_p, "_jit_finishi", state_p, pointer_t)
        sig(node_p, "_jit_forward", state_p)
        sig(node_p, "_jit_indirect", state_p)
        sig(node_p, "_jit_label", state_p)
        sig(node_p, "_jit_new_node_p", state_p, code_t, pointer_t)
        sig(node_p, "_jit_new_node_pww", state_p, code_t, pointer_t, word_t, word_t)
        sig(node_p, "_jit_new_node_qww", state_p, code_t, int32_t, int32_t, word_t)
        sig(node_p, "_jit_new_node_w", state_p, code_t, word_t)
        sig(node_p, "_jit_new_node_ww", state_p, code_t, word_t, word_t)
        sig(node_p, "_jit_new_node_www", state_p, code_t, word_t, word_t, word_t)
        sig(node_p, "_jit_note", state_p, char_p, ctypes.c_int)
        sig(pointer_t, "_jit_address", state_p, node_p)
        sig(pointer_t, "_jit_emit", state_p)
        sig(state_p, "jit_new_state")
        sig(void, "_jit_clear_state", state_p)
        sig(void, "_jit_destroy_state", state_p)
        sig(void, "_jit_ellipsis", state_p)
        sig(void, "_jit_epilog", state_p)
        sig(void, "_jit_finishr", state_p, gpr_t)
        sig(void, "_jit_getarg_i", state_p, gpr_t, node_p)
        sig(void, "_jit_getarg_l", state_p, gpr_t, node_p)
        sig(void, "_jit_link", state_p, node_p)
        sig(void, "_jit_patch", state_p, node_p)
        sig(void, "_jit_patch_at", state_p, node_p, node_p)
        sig(void, "_jit_prepare", state_p)
        sig(void, "_jit_prolog", state_p)
        sig(void, "_jit_pushargi", state_p, word_t)
        sig(void, "_jit_pushargr", state_p, gpr_t)
        sig(void, "_jit_ret", state_p)
        sig(void, "_jit_reti", state_p, word_t)
        sig(void, "_jit_retr", state_p, gpr_t)
        sig(void, "_jit_retval_c", state_p, gpr_t)
        sig(void, "_jit_retval_i", state_p, gpr_t)
        sig(void, "_jit_retval_s", state_p, gpr_t)
        sig(void, "_jit_retval_uc", state_p, gpr_t)
        sig(void, "_jit_retval_us", state_p, gpr_t)
        sig(void, "finish_jit")
        sig(void, "init_jit", ctypes.c_char_p) # NOTE: Don't use char_p

        if wordsize == 64:
            sig(void, "_jit_retval_l", state_p, gpr_t)
            sig(void, "_jit_retval_ui", state_p, gpr_t)