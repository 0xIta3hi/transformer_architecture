## Purpose
- this is me trying to write the complete transformer architecture from scratch without using any AI tools at all, completely from scratch. No github-copilot, not even simple google, just me my brain the "Attention is all you need" paper, neovim and that's it.
- I hope to complete it by the end of the month ie in 13 days. while preparing to leave for my new hostel. let's see how it comes up. 
- Purpose for doing this is to understand deeply how an LLM function, by doing this i believe that i will also be able to grasp the concept of AI vulnerabilities much faster and understand them truely not just another script kiddie, an actual AI security reseacher.
- And that's the whole point in this. 
- wish me luck, let's see if i can pull this off.

## What are my learning:
### Self-attention 
- This is the way to predict next token in the output generation. 
- the input is divided into 3 things, namely query (q), key(k), and value (v)
- the query is the part of the input which proposes a question like what am i looking for? eg, am i looking for a noun,verb,etc?
- Key is the part which gives the context as in who am i? eg. i am a vovel.
- value is the actual meaning of token
- to have the "attention" we first convert all these into a d_k dimensioned vector. Once done, we then firstly matrix multiply the Q and the transpose of K matrix.
- once done, we then divide it with the root of d_k and apply softmax to get all the numbers in the range of 0 to 1
- then we multiple the whole thing with the Value matrix. and the output we get is our attention
