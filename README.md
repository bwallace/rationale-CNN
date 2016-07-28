# rationale-CNN
A [Keras](http://keras.io/) implementation of our CNNs with "rationales". Reference article: [https://arxiv.org/abs/1605.04469](https://arxiv.org/abs/1605.04469).

![model schematic](https://raw.githubusercontent.com/bwallace/rationale-CNN/master/figures/rationale-CNN-figure.png)

# installing

This should run fine in Python 2.7 or 3.x. Requires the usual stack of numpy/scipy/pandas, plus an up-to-date version of [Keras](http://keras.io/); we developed and test on 1.0.5. 

# configuring

Your data should be formatted as a CSV with no headers. Each row is expected to have the following format: 

`doc_id, doc_lbl, sentence_number, sentence, sentence_lbl`

Where `doc_id` is an arbitrary unique document identifier; `doc_lbl` is a (binary) label on the *document*; `sentence_number` is the index of the sentence within a document (each document starting at sentence 0); and `sentence_lbl` encodes whether the sentence is a rationale (1) or not (-1). The *movies.txt* file under the `data' directory provides a concrete example.

You will need to create a simple config.ini file, which points to your data. In particular, it should look like this:

`[paths]`  
`data_path=/Path/to/data.txt`  
`word_vectors_path=/Path/to/embeddings.bin`  

The embeddings should contain the pre-trained word vectors to use for initialization (these will be tuned).

# running on the `movies` dataset

As an example, we distribute the *movies* dataset, which is from Zaidan's original work on [rationales](http://www.cs.jhu.edu/~ozaidan/rationales/). To run this, create a movies_config.ini file, with the above two entries. We suggest using the standard [Google news trained word2vec embeddings](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjD2fGy2f_NAhXs54MKHRdcD9EQFggcMAA&url=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F0B7XkCwpI5KDYNlNUTTlSS21pQmM%2F&usg=AFQjCNF9AQjAMpwC_OiLOOrdEvZC2Y3NSw&sig2=7mcbKV9x-ApwMB8IWwym9Q&bvm=bv.127521224,d.amc).

`THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_RA_CNN.py --inifile=/path/to/movies_config.ini --sentence-epochs=15 --batch-size=50 --document-epochs=200 --dropout-document=0.5 --dropout-sentence=0.7 --name=movies --mf=25000 --max-doc-length=40 --max-sent-length=20 --shuffle --val-split=.1 --num-filters=20`

Obviously, the inifile path needs to be changed accordingly. Note that in practice, one needs to tune the sentence dropout on the train set, because the model is rather sensitive to this hyper-parameter. 

For explanations regarding all possible arguments, use:

`python train_RA_CNN.py -h`

# acknowledgements & more info

This work is part of the [RobotReviewer](https://robot-reviewer.vortext.systems/) project, and is generously supported by the National Institutes of Health (under the National Library of Medicine), grant R01-LM012086-01A1. 



