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
I used python's natural language toolkit called nltk to get the part of speech(POS) tags of each word in a sentence. Then I built a dictionary mapping each POS tag to an index. I processed each sentence and generated a word vector of the POS tags. This word vector is given as input to the neural network which is then trained using a stochastic gradient descent. 

The cost was jumping at later epochs, so I used a variable learning rate updating the learning rate by a factor of 0.9999 after each epoch. The best classification rate observed is 0.91.
