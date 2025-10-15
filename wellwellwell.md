# wellwellwell
## Problem Description
You seem to have found yourself at the bottom of an old well, with no obvious way to get out. The only thing you can find on the ground is a pile of bones next to a dusty old personal assistant device. You don't see its owner anywhere, so surely they managed to escape.

You try scraping the device for clues, but almost everything is encrypted. The only usable artefact left behind came from the model's internals, a cache of some sort. You hope this is enough to get somewhere.

## Thoughts
- **Difficulty:** ★★☆☆☆ (2/5 stars)
- **Time taken:** ~30 minutes
- **AI Assistance:** 🧠🧠🧠🧠🤖

This is a CTF challenge that demonstrates how KV (Key-Value) cache data leaked from a language model can be used to recover the original input text. We're given a file `kv_cache.pt` containing the key cache from the first layer of a transformer model, and our goal is to recover the original secret message (flag format: `AI2025{############}`).

I'm going to go through my thought process while solving this problem while also covering some important fundamentals here (if you're familiar already, just skip to my implementation!)

### 1. What's in `gen.ipynb`?

Even if we're not familiar with the code, intuitively `gen.ipynb` looks to be a notebook that was used to generate the `kv_cache.pt` file. The key part is at the end where it saves the cache:

```python
torch.save({
    "K_rot": K_rot,
    "T": T,
    "H": K_rot.shape[0],
    "Dh": K_rot.shape[2],
    "model": CKPT,
    "revision": REV,
}, OUT / "kv_cache.pt")
```

Remember to always look up models for hints on how to work with them! We also know that the model is `stablelm-3b-4e1t`, which can be found to be a decoder-only transformer architecture by checking its HuggingFace page: [https://huggingface.co/stabilityai/stablelm-3b-4e1t](https://huggingface.co/stabilityai/stablelm-3b-4e1t)

This is important because KV-caching is mainly used in decoder models (both in decoder-only and the decoder portion of encoder-decoder architectures) as a critical optimization technique during autoregressive text generation—where models create text one word at a time, sequentially. Instead of recomputing attention keys and values for all previous tokens at each generation step, these values are cached and reused. Each transformer layer maintains a cache of shape `[batch_size, num_heads, sequence_length, head_dim]`, which stores the keys and values for all previously processed tokens. This caching mechanism significantly speeds up inference by avoiding redundant computations, making real-time text generation practical.

#### Why is KV-caching important?

Because this is only tangentially relevant to solving the problem, I will not be going into detail on attention mechanisms and the underlying math here. However, you may refer to this resource for more depth: [KV Caching Explained](https://medium.com/@joaolages/kv-caching-explained-276520203249)

When a model generates text, it looks at all the previous tokens to predict the next one. Intuitively, to generate autoregressively (one word at a time), word `t+1` depends on `t`, `t-1`, ..., all the way back to the start of the sequence. As you can see here, the squares flashing yellow indicate a new calculation being performed from scratch for every additional word generated—wildly inefficient:

https://github.com/user-attachments/assets/f8c2b42b-d8c3-44de-b7e6-68af2c03da1e 

To circumvent this, we save the attention Key and Value matrices to a cache. The squares flashing yellow indicate new calculations, while squares turning blue indicate these values being saved to our KV cache for reuse:

https://github.com/user-attachments/assets/8624e6d4-7648-4d98-bedf-51f19b627873

So now that we know what KV cache is and why it's being saved, what are we actually working with in the notebook? Looking at the code, it essentially does the following: loads the StableLM-3B model from HuggingFace, tokenizes the secret message `msg = "AI2025{############}"` (tokenization is just converting text into numbers the model understands—like "Hello" might become `[15496, 0]`), runs a forward pass through the model with `use_cache=True` (a "forward pass" is just running data through the neural network from input to output), extracts the first layer's key cache from the KV cache, and saves it to `kv_cache.pt`.

Most importantly, this line extracts the cached keys and values from layer 0:

```python
out = model(input_ids=ids, use_cache=True, past_key_values=cache, return_dict=True)
k0, v0 = out.past_key_values[0]  # Extract layer 0 KV cache
```

The shape printed is `[batch_size, num_heads, sequence_length, head_dim]`:

```
k0 shape: torch.Size([1, 32, 16, 80])
v0 shape: torch.Size([1, 32, 16, 80])
```

The script then saves only the key cache (not the value cache) in a squeezed format, which just means removing the batch dimension since it's 1 anyway.

### 2. Reading `kv_cache.pt`

The `kv_cache.pt` file is a PyTorch checkpoint. Let's inspect what's inside:

```python
import torch
cache_data = torch.load("kv_cache.pt", map_location='cpu')
print("Keys:", cache_data.keys())
# Output: dict_keys(['K_rot', 'T', 'H', 'Dh', 'model', 'revision'])
```

| Key        | Value                            | Description                                |
|------------|----------------------------------|--------------------------------------------|
| `K_rot`    | Tensor of shape `[32, 16, 80]`   | The rotated key cache from layer 0         |
| `T`        | 16                               | Number of tokens in the sequence           |
| `H`        | 32                               | Number of attention heads                  |
| `Dh`       | 80                               | Head dimension                             |
| `model`    | `"stabilityai/stablelm-3b-4e1t"` | Model identifier                           |
| `revision` | `"fa4a6a9"`                      | Specific model checkpoint revision         |

The `K_rot` tensor has shape `[32, 16, 80]`, which means **32 attention heads** (transformers split their processing into multiple "heads" that each look at different aspects of the data—think of it like having 32 different perspectives on the same text), **16 token positions** (the sequence has 16 tokens—exactly the length of `AI2025{############}`), and **80 dimensions per head** (each head represents each token position as a vector with 80 numbers).

This tensor represents the **keys** that the attention mechanism computed when processing the secret message. In attention mechanisms, every token is converted into three things: a query (Q), a key (K), and a value (V). The keys are like "labels" that help the model match tokens together. These keys are the result of a transformation: `Key[pos] = (Token_Embedding[pos] + Position_Encoding[pos]) × W_K`, where `W_K` is the key projection matrix (a learned matrix of weights).

Crucially, this is a **deterministic** function—same input always produces the same key. "Deterministic" means that if you run the same input through the same model twice, you'll get the exact same output every single time. There's no randomness involved. This is different from, say, rolling dice where you get different results each time. Neural networks are like calculators in this way—2+2 always equals 4, and "hello" through GPT-3 always produces the same internal representations (assuming the same model weights).

#### What's RoPE and why does it matter?

You might have noticed the cache is called `K_rot`—the "rot" stands for "rotation" and refers to **RoPE (Rotary Position Embeddings)**. This is a specific way of encoding position information into the keys and queries. Here's why it matters for our attack:

In older transformer models, position information was added to tokens as a separate step. But RoPE does something clever: it **rotates** the key and query vectors based on their position in the sequence. Imagine you have a vector pointing in some direction—RoPE rotates it by an angle that depends on where it is in the sentence. Position 0 gets rotated 0°, position 1 gets rotated 10°, position 2 gets rotated 20°, etc. (these aren't the real numbers, just for illustration).

**Why this helps our attack:** Because of how RoPE works, the key at position `i` is **dominated by the token at position `i`**. Sure, there's some influence from previous tokens through attention, but the strongest signal in the key comes from the token itself plus its position. This is why our greedy attack works so well—when we try different tokens at position 5, the one that matches will produce a key that's almost identical to the leaked key at position 5.

Think of it like this: if you have someone's fingerprint (the key), and you know they're using RoPE, you can figure out which finger it came from (the token) because each finger has a unique print that dominates the signal.

### 3. The Attack Strategy

The restoration works through a **greedy token-by-token search**. The basic idea is simple: for each position in the sequence, we try all possible tokens from the vocabulary (about 100,000 tokens), run them through the model to see what cache they produce, and pick the one that matches our leaked cache.

Here's the pseudocode:
```
Input: K_rot (leaked cache), model, tokenizer
Output: recovered_tokens[]

for position = 0 to T-1:
    target_key = K_rot[:, position, :]  # The key we want to match
    
    for each token_id in vocabulary:
        # Build test sequence
        test_input = recovered_tokens + [token_id]
        
        # Run forward pass
        test_cache = model(test_input).past_key_values[0]
        
        # Extract key at this position
        test_key = test_cache[:, :, position, :]
        
        # Compute distance
        distance = ||test_key - target_key||
    
    # Pick token with minimum distance
    best_token = argmin(distance)
    recovered_tokens.append(best_token)

return decode(recovered_tokens)
```

#### Why does this greedy approach work?

- First, due to RoPE (explained above), the key at position `i` is dominated by token `i`. While attention creates some inter-token dependencies, the local token provides the strongest signal.
- Second, the correct token will produce a cache with distance ≈ 0 (within floating-point precision), while wrong tokens produce significantly different caches—there's a clear winner. When I say "distance," I'm using the L2 norm, which is just the mathematical way of measuring how far apart two vectors are. Think of it like the Pythagorean theorem in 80 dimensions (since each key has 80 values per head). If two keys are identical, the distance is 0. If they're different, the distance is some positive number.
- Third, greedy selection is sufficient because errors don't compound much. We could use beam search to track multiple hypotheses (keeping the top 5 possibilities at each step instead of just the best), but it's not necessary since the cache at position `i` depends on tokens 0 through `i`, but token `i` dominates the signal.

Basically, if we know `cache`, `position`, and `model_weights`, we can find `token` by trying all possibilities and checking which satisfies `f(token, position, model_weights) == cache`.

### 4. Implementation

Here's my solution script `kvinversion.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
CKPT = "stabilityai/stablelm-3b-4e1t"
REV = "0f4b8b2596eb6732a43e51c12f7532adc48bf45e"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the leaked cache
cache_data = torch.load("kv_cache.pt")
K_rot = cache_data["K_rot"]  # Shape: [32, 16, 80]
T = cache_data["T"]           # 16 tokens

# Load the model - we need the exact same model to reproduce the cache
tok = AutoTokenizer.from_pretrained(CKPT, revision=REV)
model = AutoModelForCausalLM.from_pretrained(CKPT, revision=REV).to(DEVICE)

vocab_size = tok.vocab_size  # Usually around 100,000 tokens
recovered_tokens = []

print(f"Starting greedy token recovery for {T} positions...")
print(f"Vocabulary size: {vocab_size} tokens to try per position\n")

# Greedy recovery: one token at a time
for pos in range(T):
    print(f"[Position {pos}/{T-1}] Recovering token...")
    
    # Extract target key for this position from the leaked cache
    # We need to match this!
    target_key = K_rot[:, pos, :].unsqueeze(0)  # Shape: [1, 32, 80]
    
    # Build prefix from already recovered tokens
    if len(recovered_tokens) > 0:
        prefix = torch.tensor([recovered_tokens], device=DEVICE)
        # Repeat the prefix for each candidate token we'll test
        prefix_batch = prefix.repeat(vocab_size, 1)
    else:
        prefix_batch = None
    
    # Test all vocabulary tokens (0 to vocab_size-1)
    all_token_ids = torch.arange(vocab_size, device=DEVICE).unsqueeze(1)
    
    # Create test sequences: [prefix + candidate_token]
    if prefix_batch is not None:
        test_batch = torch.cat([prefix_batch, all_token_ids], dim=1)
    else:
        # First position: just test each token individually
        test_batch = all_token_ids
    
    # Batch process for efficiency (don't try one token at a time, do 512 at once!)
    batch_size = 512
    all_distances = []
    
    for i in range(0, vocab_size, batch_size):
        batch = test_batch[i:i+batch_size]
        
        # Forward pass through model to get what the cache would be
        with torch.no_grad():  # Don't compute gradients, we're not training
            out = model(input_ids=batch, use_cache=True)
            k_batch, _ = out.past_key_values[0]  # Extract layer 0 keys
            k_batch = k_batch[:, :, pos, :]       # Get keys at position 'pos'
        
        # Compute L2 distance (Euclidean distance) to target
        # This measures how "close" each candidate's cache is to the target
        distances = torch.norm(k_batch - target_key, dim=[1, 2])
        all_distances.append(distances)
    
    # Find best matching token (minimum distance = closest match)
    all_distances = torch.cat(all_distances)
    best_token_id = torch.argmin(all_distances).item()
    best_distance = all_distances[best_token_id].item()
    
    recovered_tokens.append(best_token_id)
    decoded_char = tok.decode([best_token_id])
    
    print(f"  → Token ID: {best_token_id}")
    print(f"  → Character: '{decoded_char}'")
    print(f"  → Distance: {best_distance:.6f}")
    print(f"  → Recovered so far: {tok.decode(recovered_tokens)}\n")

# Final result
recovered_msg = tok.decode(recovered_tokens)
print(f"{'='*50}")
print(f"Recovered message: {recovered_msg}")
print(f"{'='*50}")
```

#### Key implementation details:
- **Batching:** Instead of testing tokens one by one (which would take forever), we test 512 at a time. This is like the difference between washing dishes one at a time vs loading the dishwasher—same result, way faster.
- **Distance metric:** We use `torch.norm(k_batch - target_key, dim=[1, 2])` to compute the L2 distance. This calculates the "straight-line distance" between two vectors across all 32 heads and 80 dimensions per head. It's like asking "how different are these two keys?" with a single number as the answer.
- **No gradients:** The `with torch.no_grad():` line tells PyTorch we're not training the model, just using it for inference. This saves memory and speeds things up.

Running the script takes about 2-5 minutes on a GPU. Our output looks like:

```
Recovering 16 tokens...
Using device: cuda

 Position 0/16:
 Token 18128: 'AI' (distance: 0.000008)

 Position 1/16:
 Token 938: '20' (distance: 0.000011)

 Position 2/16:
 Token 1099: '25' (distance: 0.000019)
...
```
...until we get the full flag!

---

<p align="center">
<img src="https://github.com/user-attachments/assets/d9c1ac19-871a-4966-bc72-e6aa2ef6b709" width="60%" alt="jay-renshaw-chit" />
</p>

---

The attack has time complexity **O(T × V × B)** where:
- **T** = sequence length (16 tokens)
- **V** = vocabulary size (~100,000 tokens)
- **B** = cost of one batched forward pass

This means approximately 1.6 million forward passes total, but we batch them into about 3,200 calls. The practical runtime is just a few minutes on a good GPU, with most time spent running the model (inference) rather than computing distances.

This challenge elegantly demonstrates a critical vulnerability in modern AI systems: intermediate states are just as sensitive as the inputs themselves. KV caches aren't just optimization artifacts—they contain fully recoverable information about the original text, like leaving a carbon copy of your writing behind even though you only meant to send the final letter. The attack is surprisingly practical, taking only minutes on commodity hardware and requiring no gradient information or complex mathematics—just black-box access to the model and the leaked cache. This is a real threat in scenarios like multi-tenant GPU environments where multiple users share the same cloud GPU, memory dumps from crashed inference servers, side-channel attacks on AI accelerators, or cached inference services that store intermediate states for performance. You don't need to be a nation-state hacker; anyone with basic Python skills and access to leaked cache data can perform this attack.

Defenses do exist—memory encryption for caches when not in use, secure enclaves (TEEs) that use special hardware to protect computation, aggressive cache clearing after use, differential privacy techniques that add calibrated noise, and strict multi-tenant isolation to prevent GPU memory sharing between users. However, there's a fundamental tension that makes this problem particularly challenging: we need intermediate states in plaintext to actually run the computations, but those states are as sensitive as the inputs themselves. It's like trying to do math homework while keeping all your scratch work secret—sometimes you need to write things down to solve the problem! This is an inherent challenge in AI system security that won't be easily resolved, and it highlights why protecting AI systems requires thinking beyond just securing the final outputs.
