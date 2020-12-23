# dont forget to add first line as "num_of_words emb_dim" for umwe (sed command for inserting line to the head of file)

#./fasttext print-word-vectors ../cc.en.128.bin < ../wiki_wordsOnly/wiki.en.128.vec > ../wiki_vectors_128/wiki.en.vec
#./fasttext print-word-vectors ../cc.es.128.bin < ../wiki_wordsOnly/wiki.es.128.vec > ../wiki_vectors_128/wiki.es.vec
#./fasttext print-word-vectors ../cc.fr.128.bin < ../wiki_wordsOnly/wiki.fr.128.vec > ../wiki_vectors_128/wiki.fr.vec
#./fasttext print-word-vectors ../cc.it.128.bin < ../wiki_wordsOnly/wiki.it.128.vec > ../wiki_vectors_128/wiki.it.vec
../fastText/fasttext print-word-vectors ../../vae/model_checkpoints/cc_bins_128/cc.tr.128.bin < ../wiki_wordsOnly/wiki.tr.128.vec > ../wiki_vectors_128/wiki.tr.vec
