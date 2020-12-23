
### This bash script downloads the fastText monolingual embedding vectors for:
###	[en, fr, es, it, tr] 

#curl -o wiki.en.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
#curl -o wiki.fr.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec
#curl -o wiki.es.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.es.vec
#curl -o wiki.it.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.it.vec
#curl -o wiki.tr.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tr.vec
#curl -o wiki.tr.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tr.vec
#curl -o wiki.ru.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.ru.vec
curl -o cc.fr.300.bin https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz
curl -o cc.es.300.bin https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz
curl -o cc.it.300.bin https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.bin.gz
curl -o cc.tr.300.bin https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.bin.gz
