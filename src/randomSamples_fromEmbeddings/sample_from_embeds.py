import fasttext
import random
import torch
import pickle

langs = ["fr", "it", "es", "en"]

for lang in langs:
    ft = fasttext.load_model("../model_checkpoints/cc_bins_128/cc."+lang+".128.bin")
    ft.get_dimension()
    size = 10000
    random_words = random.sample(ft.get_words(), size)
    samples = {}
    for i in range(size):
      samples[random_words[i]] = torch.from_numpy(ft.get_word_vector(random_words[i]))
    filename = "sample_fasttext_embs_"+lang+"_128.pickle"
    outfile = open(filename,'wb')

    pickle.dump(samples,outfile)
    outfile.close()

