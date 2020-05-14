 if first_idx is None:
                first_idx = np.arange(self.shape[0])
                second_idx = np.arange(self.shape[1])
                out_rows = np.array([self.perturb(x, axis=0,keep_changes=False)
                            for x in first_idx])
                out_cols = np.array([self.perturb(x, axis=1,keep_changes=False)
                            for x in second_idx])
                return out_rows, out_cols
            elif second_idx is None:
                output = np.array([self.perturb(x, axis=axis,keep_changes=False)
                            for x in first_idx])
                return output
            else:
                out_rows = np.array([self.perturb(x, axis=0,keep_changes=False)
                            for x in first_idx])
                out_cols = np.array([self.perturb(x, axis=1,keep_changes=False)
                            for x in second_idx])
                return out_rows, out_cols
        ################################################################
        if first_idx is None:
            if second_idx is not None:
                 raise ValueError("There need to be a value for first_idx if a value for 
                 second_idx is supplied")
            else:
                if bipartite_network:
                    first_idx = np.arange(self.shape[0])
                    second_idx = np.arange(self.shape[1])
                    out_rows = [self.perturb(x, axis=0,keep_changes=False,
                                             bipartite_network=bipartite_network)
                                for x in first_idx]
                    out_cols = [self.perturb(x, axis=1,keep_changes=False,
                                             bipartite_network=bipartite_network)
                                for x in second_idx]
                    return out_rows, out_cols
                else:
                    first_idx = np.arange(self.shape[0])
                    output = [self.perturb(x,keep_changes=False,
                                           bipartite_network=bipartite_network)
                              for x in first_idx]
                    return output
        else:
            if bipartite_network:
                if second_idx is None:
                    output = [self.perturb(x, axis=axis, keep_changes=False,
                                           bipartite_network=bipartite_network)
                              for x in first_idx]
                    return output
                else:
                    out_rows = [self.perturb(x, axis=0,keep_changes=False,
                                             bipartite_network=bipartite_network)
                                for x in first_idx]
                    out_cols = [self.perturb(x, axis=1,keep_changes=False,
                                             bipartite_network=bipartite_network)
                                for x in second_idx]
                    return out_rows, out_cols
            else:
                output = [self.perturb(x,keep_changes=False,
                                       bipartite_network=bipartite_network)
                          for x in first_idx]
                return output
        ###############################################################################
        # Order: most likely or important case first
        if first_idx is None:
            if second_idx is not None:
                raise ValueError("There need to be a value for first_idx if a value for 
                second_idx is supplied")
            elif not bipartite_network:
                first_idx = np.arange(self.shape[0])
                output = [self.perturb(x,keep_changes=False,
                                   bipartite_network=bipartite_network)
                          for x in first_idx]
            else:
                first_idx = np.arange(self.shape[0])
                second_idx = np.arange(self.shape[1])
                out_rows = [self.perturb(x, axis=0,keep_changes=False,
                                         bipartite_network=bipartite_network)
                            for x in first_idx]
                out_cols = [self.perturb(x, axis=1,keep_changes=False,
                                         bipartite_network=bipartite_network)
                            for x in second_idx]
                return out_rows, out_cols
#######################################################################
if not self.bipartite_network:
    if idx1==[]:
        raise ValueError("idx1 can not be empty for bipartire_network=False")
    if idx2 is not None:
        warnings.warn("Indices in idx2 ignored, changing only indices in idx1")
    if idx1 is None:
        idx1 = np.arange(self.shape[0])
    output = np.array([self._method(x,axis=0, keep_changes=False,
                                    bipartite_network=self.bipartite_network)
                       for x in idx1])
    return output
else:
    if idx1==[] and idx2==[]:
        raise ValueError("There has to be indices to change in either idx1 or idx2")
    if idx1==None:
        idx1 = np.arange(self.shape[0])
    if idx2==None:
        idx2 = np.arange(self.shape[1])
    if idx1!=[]:
        out_rows = np.array([self._method(x, axis=0,keep_changes=False,
                                          bipartite_network=self.bipartite_network)
                             for x in idx1])
        if idx2==[]:
            return out_rows
    if idx2!=[]:
        out_cols = np.array([self._method(x, axis=1,keep_changes=False,
                                          bipartite_network=self.bipartite_network)
                             for x in idx2])
        if idx1==[]:
            return out_cols
    return out_rows, out_cols
