import torch
from cs336_basics.softmax import softmax

def decode(model, x, eot_token_id, max_length, temp=None, top_p=None, device=None):
    with torch.no_grad():
        while x[-1] != eot_token_id and x.shape[-1] < max_length:
            # Get logits
            logits = model(x) # Shape (context_length, vocab_size)

            # Get next token logits
            next_token_logits = logits[-1]

            if temp is not None:
                next_token_logits = next_token_logits / temp

            # Apply softmax
            next_token_probs = softmax(next_token_logits)

            if top_p is not None:
                sorted_next_token_probs, sorted_next_token_indices = torch.sort(next_token_probs, dim=-1, descending=True)
                cum_sum = torch.cumsum(sorted_next_token_probs, dim=-1)
                
                # In the index where cum_sum exceeds top_p, set all indices after that to 0
                index_where_exceeds = torch.nonzero(cum_sum > top_p)[0] # First element where cum_sum exceeds top_p
                sorted_next_token_probs[index_where_exceeds+1:] = 0
                sorted_next_token_probs = sorted_next_token_probs / torch.sum(sorted_next_token_probs)

            # Sample from the distribution
            next_token = torch.multinomial(sorted_next_token_probs, num_samples=1)
            x = torch.cat((x, next_token), dim=-1)

    return x


    
