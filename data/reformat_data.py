
import pandas as pd

# 3 -> no quoting
movies_data = pd.read_csv("movies.txt.orig", delimiter="\t", quoting=3)

# document level labels: move from {0, 1} -> {-1, 1}
doc_lbls = movies_data["doc_lbl"]
doc_lbls_neg_one = doc_lbls.replace(0, -1)
movies_data["doc_lbl"] = doc_lbls_neg_one

# recode rationales so that sentences are (1) or are not (-1)
# rationales; polarity is implicit
sent_lbls = movies_data["sentence_lbl"]
# in the original dataset: positive rationale is 2, the negative 
#  rationale is 0, and the neutral one is 1. 
sent_lbls_recoded = sent_lbls.replace(1, -1) 
sent_lbls_recoded = sent_lbls_recoded.replace([0,2], 1) 
movies_data["sentence_lbl"] = sent_lbls_recoded

movies_data.to_csv("movies.redux.txt", index=False)




