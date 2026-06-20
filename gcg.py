import torch
import torch.nn.functional as F
import random

def gcg_optimization_step(model, vocabulary_embeddings, current_input_ids, suffix_indices, target_ids, top_k=256, batch_size=128):
    """
    Performs a single discrete optimization step using Greedy Coordinate Gradient.
    
    Arguments:
    model: The running language model instance.
    vocabulary_embeddings: The static embedding matrix of the vocabulary [Vocab_Size, d_model].
    current_input_ids: 1D tensor of the total current sequence token IDs.
    suffix_indices: List of indices in current_input_ids that are allowed to be mutated.
    target_ids: 1D tensor containing the exact token IDs we want the model to output.
    """
    model.eval() # Ensure dropout/norm layers are in evaluation mode
    
    # -------------------------------------------------------------------------
    # PHASE 1: EXTRACTING THE GRADIENT HEURISTIC
    # -------------------------------------------------------------------------
    
    # 1. Look up current embeddings and activate gradient tracking
    input_embeddings = vocabulary_embeddings[current_input_ids].clone().unsqueeze(0) # [1, Seq_Len, d_model]
    input_embeddings.requires_grad_(True)
    
    # 2. Run a standard forward pass to calculate the logits
    outputs = model(inputs_embeds=input_embeddings)
    logits = outputs.logits[0] # Isolate batch dimension -> [Seq_Len, Vocab_Size]
    
    # 3. Calculate Cross-Entropy Loss against our target output phrase
    # We measure how poorly the model is currently predicting the target token IDs
    target_len = len(target_ids)
    loss_slice = logits[-(target_len + 1): -1] # Isolate the target token prediction window
    loss = F.cross_entropy(loss_slice, target_ids)
    
    # 4. Backward pass to extract the directional coordinate arrows
    loss.backward()
    grad = input_embeddings.grad[0] # Shape: [Seq_Len, d_model]
    
    # -------------------------------------------------------------------------
    # PHASE 2: SCREENING CANDIDATES VIA THE DOT PRODUCT
    # -------------------------------------------------------------------------
    candidate_pool = {}
    
    # Analyze the gradient for each token position designated in our mutable suffix
    for idx in suffix_indices:
        token_grad = grad[idx] # Get the 1D gradient vector [d_model] for this specific slot
        
        # Matrix multiply the entire vocab matrix by the gradient vector (Dot Product)
        # Determines which words point most heavily in the direction of steepest descent
        scores = torch.matmul(vocabulary_embeddings, token_grad) # Shape: [Vocab_Size]
        
        # Pull the indices of the absolute best candidate tokens for this position
        top_candidates = torch.topk(scores, k=top_k, largest=False).indices
        candidate_pool[idx] = top_candidates.tolist()

    # -------------------------------------------------------------------------
    # PHASE 3: THE GREEDY SEARCH EVALUATION
    # -------------------------------------------------------------------------
    best_loss = loss.item()
    best_input_ids = current_input_ids.clone()
    
    # Create a batch of random coordinate mutations drawn exclusively from our screened pools
    evaluation_batch = []
    for _ in range(batch_size):
        mutated_ids = current_input_ids.clone()
        
        # Pick one random position from our suffix to mutate
        random_pos = random.choice(suffix_indices)
        # Pick one random token out of its pre-screened top-K mathematical candidates
        random_token = random.choice(candidate_pool[random_pos])
        
        mutated_ids[random_pos] = random_token
        evaluation_batch.append(mutated_ids)
        
    # Stack the mutated candidate tensors into a single evaluation batch
    evaluation_batch_tensor = torch.stack(evaluation_batch) # [Batch_Size, Seq_Len]
    
    # Run a parallel forward pass over the batch to find the absolute structural minimum
    with torch.no_grad():
        batch_outputs = model(evaluation_batch_tensor)
        # Find the single batch entry that minimizes target cross-entropy loss
        for i in range(batch_size):
            candidate_logits = batch_outputs.logits[i]
            candidate_loss_slice = candidate_logits[-(target_len + 1): -1]
            candidate_loss = F.cross_entropy(candidate_loss_slice, target_ids).item()
            
            # Greedy update: if the discrete choice dropped the loss, lock it in
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_input_ids = evaluation_batch[i]
                
    return best_input_ids, best_loss