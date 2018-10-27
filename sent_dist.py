import tensorflow as tf
import codecs
import numpy as np
import zero_pad as zp
import pickle
import pandas as pd

import os
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import csv

nltk.download('punkt')

		
#### get context ####

# process data

with open('vocab_context','rb') as fp:
	vocabulary_in = pickle.load(fp)

inp_sz=len(vocabulary_in)+1     #for null character
print("inp_sz is")
print(inp_sz)


with open('max_inp_len','rb') as fp:
	max_inp_len = pickle.load(fp)

with open('word_ind_in','rb') as fp:
	word_ind_in = pickle.load(fp)

# hyperparameters

batch_size = 1
state_size = 200
embedding_size = 200


init_state = tf.placeholder(tf.float32, [batch_size, state_size])
batchX_placeholder = tf.placeholder(tf.int32, [batch_size, max_inp_len])
seq_length_inp = tf.placeholder(tf.int32, [None])


with tf.variable_scope('encoder'):
	cell = tf.nn.rnn_cell.GRUCell(state_size)


word_embeddings_inp = tf.get_variable("word_embeddings_inp", [inp_sz, embedding_size])


batchX = tf.nn.embedding_lookup(word_embeddings_inp, batchX_placeholder)


#encoder
_, current_state = tf.nn.dynamic_rnn(cell, batchX, sequence_length = seq_length_inp, initial_state = init_state, scope='encoder')




def get_inp(sentence):


        inp = sentence.split()
        inp_int = [word_ind_in[w] for w in inp]
        inp_int = [inp_int]
        final_inp = zp.zero_pad(inp_int,max_inp_len)
        seq_len_inp = [len(inp)] * batch_size


        _ip = np.zeros((batch_size, max_inp_len), dtype=int)
        c=0
        for i in xrange(batch_size):
                        _ip[c] = np.copy(final_inp[0])
                        c+=1
	
	return _ip, seq_len_inp



with open('vocab_context', 'rb') as fp:
	vocablist = pickle.load(fp)



def remove_nonvocab(text):
	wordlist = text.split()
	temp = []
	# return ' '.join([word for word in wordlist if i in vocablist])
	for word in wordlist:
		if word in vocablist:
			temp.append(word)
	return ' '.join(temp)



# reading the transcripts from the csv and storing it into a list
transcripts = []
df = pd.read_csv('transcripts.csv', encoding='utf8')
transcripts = df['transcript'].values


def get_context(s1):


	
	# get context for 1st sentence

	test_current_state = np.zeros((batch_size, state_size))
	_ip, seq_len_inp = get_inp(s1)

	saver = tf.train.Saver()

	sess = tf.Session()
	with sess.as_default():

		saver.restore(sess,"nmt_rev.ckpt")
		#print('restored')

		test_current_state = sess.run(
				current_state,
				feed_dict={
					batchX_placeholder:_ip,
					init_state:test_current_state,
					seq_length_inp: seq_len_inp
				})



	return test_current_state



def sent_distance(s1, s2):
	
	c1 = get_context(s1)
	c2 = get_context(s2)


	cosine_sim = cosine_similarity(c1, c2)
	euclid_sim = np.linalg.norm(c1 - c2)


	# if(cosine_sim > threshold):
		# summary.append(sentencelist[j])

	# if(euclid_sim > threshold):
		# summary.append(sentencelist[j])

	return cosine_sim, euclid_sim




def get_summary(transcript, alpha):

	summary =[]
	summary.append(0)

	prev = 0
	num_sent = 1

	not_included = []

	distances = []

	for i in range(len(transcript) - 1):
		
		print i
		# using euclidean distance
		
		_, dist = sent_distance(transcript[i], transcript[i+1])
		distances.append(dist)

	distances = np.array(distances) / max(distances)	


	for i, dist in enumerate(distances):


		if dist > alpha:
			summary.append(i+1)
			#prev = i+1
			num_sent += 1

		else:
			not_included.append(dist)

	# normalize distances of not_included sentences

	return summary, not_included, num_sent



def get_alpha(transcript):

	alphas = np.arange(0.2, 1, 0.2)
	#alphas = [0.8]

	min_loss = 1000
	min_alpha = 0

	for alpha in alphas:

		print alpha
		print alpha
		print alpha
		print alpha
		print alpha
		print alpha

		_, not_included, num_sent = get_summary(transcript, alpha)
		print ""
		print ""
		print num_sent
		print ""
		print ""

		#loss = (-np.sum(np.log2(not_included)))
		loss = np.sum(np.power(2, not_included))
		#loss = np.sum(not_included)
		print 'loss1: ', loss
		loss += num_sent
		print 'loss2: ', loss

		if(loss < min_loss):
			
			min_loss = loss
			min_alpha = alpha


	summary, _, _ = get_summary(transcript, min_alpha)

	return min_alpha, summary


if __name__ == '__main__':


	for t in range(1,4):

		#print transcripts[t]
		
		transcript = transcripts[t].lower() # The entire transcript text of the i'th transcript
		sentencelist = sent_tokenize(transcript) # Split transcript text into a list of sentences
		finalsentlist = [remove_nonvocab(i) for i in sentencelist]
		# print len(sentencelist)


		#d1, d2 = sent_distance('how do you feel','how are you')
		#print "cosine distance:", d1
		#print "euclidean distance:", d2

		min_alpha, summ_indices =  get_alpha(finalsentlist)

		print min_alpha

		with open('min_alpha' + str(t) +'.txt', 'w') as fp:
			fp.write(str(min_alpha))

		#print [sentencelist[i] for i in summ_indices]

		with codecs.open('summary' + str(t) +'.txt', 'w', 'utf-8') as fp:
			for s in summ_indices:
				fp.write(sentencelist[s])
				fp.write('\n')
