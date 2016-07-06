# rationale-CNN
A [Keras](http://keras.io/) implementation of our CNNs with "rationales". Reference article: [https://arxiv.org/abs/1605.04469](https://arxiv.org/abs/1605.04469).

![model schematic](https://raw.githubusercontent.com/bwallace/rationale-CNN/master/figures/rationale-CNN-figure.png)

# configuring
Your data should be formatted as a CSV with no headers. Each row is expected to have the following format: 

`> doc_id, doc_lbl, sentence_number, sentence, sentence_lbl`

Where `doc_id` is an arbitrary unique document identifier; `doc_lbl` is a (binary) label on the *document*; `sentence_number` is the index of the sentence within a document (each document starting at sentence 0); and `sentence_lbl` encodes whether the sentence is a rationale (1) or not (-1).

You will need to create a simple config.ini file, which points to your data. In particular, it should look like this:

<code>
[paths]  
data_path=/Path/to/data.txt  
word_vectors_path=/Path/to/embeddings.bin  
</code>



