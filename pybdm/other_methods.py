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
