import math

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    Linear warmup, then cosine annealing, then constant minimum.
    """
    if t < T_w:
        return t * alpha_max / T_w
    elif t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w))) * (alpha_max - alpha_min)
    else:
        return alpha_min

        
        
    