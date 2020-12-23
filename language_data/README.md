here is the procedure we applied for umwe training:
1. we specified our vae's latent dim as 128 and fasttext monolingual vectors are 300
2. umwe takes training data from wiki.vec files (whitespace seperated plain text -> "word 300dVector") 
3. so we need to reduce the vectors in the wiki.vec files (seperate file for each language)
4. in fasttext besides .vec files there is .bin embedding binaries for each language. we first need to reduce binaries dimension (there is a script for that in fasttext repo)
5. after reducing the each .bin file, again we are going to use a fasttext script for extracting embedding vectors from a word-list
6. to use that script we need to split the words-list from the .vec files (just write a python script for that)
7. then create the new 128 dimensional .vec files
8. dont forget to change the hyperparam for training umwe with 128 dim embeddings
