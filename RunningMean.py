import torch as th

class RunningMeanStd:
    def __init__(self, eps=1e-4, shape=()):
        self.mean = th.zeros(shape, dtype=th.float64)
        self.var = th.ones(shape, dtype=th.float64)
        self.count = eps

    def update(self, x):
        # x: torch tensor, shape (N, *shape) or (N,)
        x = x.to(dtype=th.float64)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        # Welford style batch update
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + th.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x, clip=10.0):
        x = x.to(dtype=th.float64)
        std = th.sqrt(self.var + 1e-8)
        return ((x - self.mean) / std).to(dtype=th.float32)
