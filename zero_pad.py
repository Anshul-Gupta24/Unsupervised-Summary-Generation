'''
	add zero-padding to vectors
'''
import numpy as np


def zero_pad(inp, max_sequence_length=None):
        
	sequence_lengths = [len(seq) for seq in inp]
	if max_sequence_length is None:
        	max_sequence_length = max(sequence_lengths)
        
	inputs_batch_major = np.zeros(shape=[len(inp), max_sequence_length], dtype=np.int32) # == PAD
        
	for i, seq in enumerate(inp):
                for j, element in enumerate(seq):
                        inputs_batch_major[i,j] = element
        
	return inputs_batch_major, max_sequence_length


