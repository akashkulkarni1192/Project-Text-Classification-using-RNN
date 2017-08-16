Project-Text-Classification-using-RNN
=====================================
Classification of poetry lines using word vector of its POS tagging

Goal
----
To create poetry classifier model using RNN which will classify poems by identifying its poet.

Dataset
-------
Poems written by Edgar Allan and Robert Frost.

Implementation
--------------
I used python's natural language toolkit called nltk to get the part of speech(POS) tags of each word in a sentence. Then I built a dictionary mapping each POS tag to an index. I processed each sentence and generated a word vector of the POS tags. Thos word vector is given an input to the neural network.
