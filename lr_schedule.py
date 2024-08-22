from math import cos, pi
from typing import Callable

def get_schedule(warmup_steps: int, max_lr: float, target_steps: int, min_lr: float) -> Callable[[int], float]:
    # 带预热的余弦退火
    def get_lr(step: int) -> float:
        if step <= warmup_steps:
            return max_lr / warmup_steps * step
        if step <= target_steps:
            # (warmup_steps, target_steps] -> (0, pi] -cos-> (1, -1]
            # (1, -1] -> (max_lr, min_lr]
            return (cos((step - warmup_steps) / (target_steps - warmup_steps) * pi) + 1) / 2 * (max_lr - min_lr) + min_lr
        return min_lr
    return get_lr

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    steps = [i for i in range(0, 1000)]
    schedule = get_schedule(10, 6e-4, 800, 1e-4)
    lrs = [schedule(step) for step in steps]
    plt.plot(lrs)
    plt.show()
