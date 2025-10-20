# Learning Guide: Neural Machine Translation (NMT)

This document provides a structured overview of the core concepts behind Neural Machine Translation (NMT), focusing on the sequence-to-sequence architecture, recurrent neural networks, and the attention mechanism.

## Learning Objectives

* Understand the role of Recurrent Neural Networks (RNNs) in processing sequential data like text.
* Grasp the fundamentals of the sequence-to-sequence (seq2seq) architecture, including the Encoder and Decoder components.
* Learn how the Attention Mechanism solves the bottleneck of the basic seq2seq model.
* Differentiate between Global and Local Attention.
* Explore different Decoding Strategies used to generate the final translation.

---

## 1. The Foundation: Sequence-to-Sequence (Seq2Seq) Architecture

At its heart, NMT treats translation as a problem of transforming one sequence (the source sentence) into another (the target sentence). The standard architecture for this is **seq2seq**.

A seq2seq model consists of two main components, both typically built using Recurrent Neural Networks (RNNs):

### a. The Encoder

The Encoder's job is to read the source sentence, word by word, and compress its meaning into a fixed-size numerical representation.

* **Process:** It takes one word (or token) from the source sentence at each timestep, processes it, and updates its internal "hidden state."
* **Hidden State:** This is a vector that captures the information from the words seen so far. The hidden state from the previous step is combined with the input of the current step.
* **Final Output:** After processing the entire source sentence, the Encoder's final hidden state is called the **Context Vector**.

### b. The Context Vector: The Bottleneck

The **Context Vector** (`c`) is a single vector that is supposed to encapsulate the entire meaning of the source sentence. In the basic seq2seq model, this is the *only* piece of information passed from the Encoder to the Decoder.

This creates a significant bottleneck:

* It's difficult to cram the meaning of a long, complex sentence into one fixed-size vector.
* The model might forget information from the beginning of the sentence by the time it reaches the end.

### c. The Decoder

The Decoder's job is to take the Context Vector and generate the target sentence, word by word.

* **Process:** It is also an RNN. It is initialized with the Encoder's Context Vector.
* At each timestep, it uses its current hidden state and the previously generated word to predict the next word in the target sentence.
* It continues generating words until it produces a special `<end-of-sentence>` token.

---

## 2. The Engine: Recurrent Neural Networks (GRU)

RNNs are the neural networks designed to work with sequences. For NMT, a specific and efficient type of RNN is often used: the **Gated Recurrent Unit (GRU)**.

A standard RNN can struggle with long sequences due to the "vanishing gradient problem," where it forgets long-term dependencies. GRUs solve this by using special "gates."

* **Update Gate (`z`):** Decides how much of the past information (previous hidden state) to keep and how much new information to add.
* **Reset Gate (`r`):** Decides how much of the past information to forget.

These gates allow a GRU to learn and selectively remember information over long sequences, making it much more powerful than a simple RNN for tasks like machine translation.

---

## 3. The Breakthrough: The Attention Mechanism

The limitation of the fixed-size context vector was the biggest hurdle for NMT. The **Attention Mechanism** was the revolutionary idea that solved it.

**Core Idea:** Instead of forcing the Decoder to rely on a *single* context vector, let's allow it to "look back" at the entire source sentence and decide which words are most relevant at each step of the translation process.

With attention, the Decoder no longer gets just the final hidden state. Instead, it gets access to **all** the hidden states produced by the Encoder (one for each source word).

At each step of generating a target word, the Decoder performs these actions:

1.  **Calculate Alignment Scores:** It compares its current hidden state with *every* hidden state from the Encoder. This produces a "score" for each source word, indicating how relevant it is to the current target word being generated.
2.  **Compute Attention Weights:** The scores are passed through a `softmax` function, which converts them into probabilities. These are the "attention weights"—they sum to 1 and show the distribution of focus.
3.  **Create a Dynamic Context Vector:** A new, dynamic context vector is created by taking a weighted sum of all the Encoder's hidden states, using the attention weights.

This dynamic context vector is unique for each target word being generated, effectively allowing the Decoder to focus on different parts of the source sentence as needed.

### Source/Target Alignments

The attention weights themselves are incredibly useful. If you visualize them in a matrix, you get a **source/target alignment**. This shows which source words the model was "paying attention to" when it generated each target word.

### Types of Attention

#### a. Global Attention

This is the mechanism described above. At each step, the Decoder considers *all* hidden states from the Encoder to compute the context vector. It's thorough but can be computationally expensive for very long sentences.

#### b. Local Attention

To improve efficiency, local attention forces the model to focus on only a small subset (a "window") of the source words.

1.  The model first predicts an aligned position `p_t` in the source sentence for the current target word.
2.  It then considers a window of source words around `p_t` (e.g., 5 words to the left, 5 to the right).
3.  The attention mechanism then works just like global attention but only over this smaller, relevant window of words.

---

## 4. The Final Step: Decoding Strategies

The Decoder outputs a probability distribution over the entire vocabulary for the next word. A decoding strategy is the algorithm we use to select a word from this distribution to build the final translation.

### a. Greedy Search

This is the simplest strategy. At each step, simply choose the word with the highest probability.

* **Advantage:** Very fast and computationally cheap.
* **Disadvantage:** Often produces suboptimal translations. An early, slightly wrong choice cannot be undone, even if it leads to a poor overall sentence.

### b. Beam Search

This is the most common and effective strategy. Instead of committing to the single best word at each step, it keeps track of the `k` most probable partial translations (called "hypotheses"). The number `k` is the **beam width**.

* **Process:**
    1.  At the first step, generate the `k` most likely first words.
    2.  For each of these `k` hypotheses, generate all possible next words and calculate the probability of the new, longer hypotheses.
    3.  From all the possible new hypotheses, select the top `k` and discard the rest.
    4.  Repeat until all `k` hypotheses end with an `<end-of-sentence>` token.
* **Result:** The final translation is the hypothesis with the highest overall probability. Beam search explores a much larger search space than greedy search and almost always produces better, more fluent translations.