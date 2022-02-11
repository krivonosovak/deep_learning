
# Skip Gram Negative Sampling Pytorch Implementation

* refer paper [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546v1)
* use CUDA
	

# Usage
	# batch_gen - batch pair (center, target) generator 
	
	VOCAB_SIZE = 9424
	VOCAB_SIZE = 9424
	EMBED_SIZE = 100  # dimension of the word embedding vectors
	NUM_SAMPLED = 5  # Number of negative examples to sample.
	LEARNING_RATE = .9
	NUM_TRAIN_STEPS = 100000
	SKIP_STEP = 2000

	model = SkipGram(VOCAB_SIZE, EMBED_SIZE)
	final_embed_matrix = train(model, batch_gen, NUM_TRAIN_STEPS, LEARNING_RATE, SKIP_STEP, NUM_SAMPLED)

	