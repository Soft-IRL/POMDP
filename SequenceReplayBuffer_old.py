import torch

class SequenceReplayBuffer:
    """
    Ring-buffer that stores single time-steps on CPU and returns
    fixed-length sequences on request.
    """

    def __init__(self, capacity, obs_shape, act_shape,
                 seq_len=8, device="cuda", obs_dtype=torch.uint8):
        self.capacity = capacity          # max *time-steps*
        self.seq_len  = seq_len           # L
        self.device   = torch.device(device)
        self.obs_dtype  = obs_dtype

        # ------------------------------------------------------------------ #
        #  storage – **CPU**, pinned so GPU transfer is DMA & non-blocking   #
        # ------------------------------------------------------------------ #
        pin = True                        # always pin; harmless on CPU
        self.obs       = torch.empty((capacity, *obs_shape),  dtype=obs_dtype, pin_memory=pin)
        self.action    = torch.empty((capacity, *act_shape),  dtype=torch.int64 , pin_memory=pin)
        self.reward    = torch.empty((capacity,),             dtype=torch.float32, pin_memory=pin)
        self.done      = torch.empty((capacity,),             dtype=torch.bool  , pin_memory=pin)
        self.step_type = torch.empty((capacity,),             dtype=torch.int64 , pin_memory=pin)

        # start-flags: valid[i] == True  ⇒  slice [i … i+L-1] is usable
        self.valid = torch.zeros(capacity, dtype=torch.bool)  # CPU

        self.ptr  = 0              # next index to write
        self.full = False

    # ---------------------------------------------------------------------- #
    # helper – recompute validity of windows that *touch* idx                #
    # ---------------------------------------------------------------------- #
    def _update_valid(self, idx):
        L = self.seq_len
        s0 = max(0, idx - L + 1)
        s1 = idx + 1
        starts = torch.arange(s0, s1, device="cpu")

        if not self.full:
            overflow = starts + L > self.ptr
            self.valid[starts[overflow]] = False

        done2 = torch.cat([self.done, self.done], dim=0)
        windows = done2.unfold(0, L - 1, 1)[starts]
        mid_done = windows.any(dim=1)
        self.valid[starts] = ~mid_done

    # ---------------------------------------------------------------------- #
    # public – add one transition                                            #
    # ---------------------------------------------------------------------- #
    def add(self, obs, action, reward, done, step_type):
        idx = self.ptr

        # write ----------------------------------------------------------------
        self.obs      [idx].copy_(torch.as_tensor(obs,  dtype=torch.uint8))
        self.action   [idx] = action
        self.reward   [idx] = reward
        self.done     [idx].copy_(torch.as_tensor(done,  dtype=torch.bool))
        self.step_type[idx] = step_type

        # update flags BEFORE moving ptr ---------------------------------------
        self._update_valid(idx)

        # advance pointer -------------------------------------------------------
        self.ptr = (idx + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    # ---------------------------------------------------------------------- #
    # public – sample a batch of sequences                                   #
    # ---------------------------------------------------------------------- #
    @torch.no_grad()
    def sample(self, batch_size: int):
        L = self.seq_len
        cap = self.capacity

        # how many elements are actually written?
        if not self.full:
            max_i = self.ptr
            if max_i < L:
                raise RuntimeError("replay buffer: not enough data yet")
            # candidate starts are [0, max_i-L]
            def draw_start():
                return int(torch.randint(0, max_i - L + 1, (1,)).item())
        else:
            # full ring: any start is possible, but must not wrap across ptr
            def draw_start():
                return int(torch.randint(0, cap, (1,)).item())

        starts = []
        tries = 0
        max_tries = batch_size * 200  # plenty

        while len(starts) < batch_size:
            s = draw_start()
            tries += 1
            if tries > max_tries:
                raise RuntimeError(
                    "replay buffer: could not find enough valid sequences "
                    "(too many episode boundaries or bug in done flags)."
                )

            # build indices for the sequence window
            if not self.full:
                idxs = torch.arange(s, s + L, device="cpu")
            else:
                idxs = (torch.arange(s, s + L, device="cpu") % cap)

                # reject sequences that wrap across the write head ptr (temporal discontinuity)
                # Equivalent: window should be a contiguous block that does NOT include ptr
                # Compute end in modulo arithmetic:
                end = (s + (L - 1)) % cap
                wraps = s > end
                if wraps:
                    continue
                if int(self.ptr) >= s and int(self.ptr) <= end:
                    continue

            # done constraint: done can only appear at the last position
            # i.e., forbid done in first L-1 slots
            if self.done[idxs[:-1]].any().item():
                continue

            starts.append(s)

        starts = torch.tensor(starts, device="cpu", dtype=torch.long)

        if not self.full:
            seq_idx = starts[:, None] + torch.arange(L, device="cpu")[None, :]
        else:
            seq_idx = (starts[:, None] + torch.arange(L, device="cpu")[None, :]) % cap

        # Gather and move to device
        obs  = self.obs[seq_idx].to(self.device, non_blocking=True)
        act  = self.action[seq_idx].to(self.device, non_blocking=True)
        rew  = self.reward[seq_idx].to(self.device, non_blocking=True)
        done = self.done[seq_idx].to(self.device, non_blocking=True)
        st   = self.step_type[seq_idx].to(self.device, non_blocking=True)

        return {"obs": obs, "action": act, "reward": rew, "done": done, "step_type": st}