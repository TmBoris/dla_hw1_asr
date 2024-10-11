echo 'downloading lm prerequisites...'
gdown https://drive.google.com/uc?id=10Y07iEqHgmSogvfkepSWsAAsa2bvuS6e
gdown https://drive.google.com/uc?id=1DzmX7ZoLcKlTLNS7b7F5Pli6FoiOwnqS
mkdir -p data/LM
mv 3-gram.pruned.1e-7.bin data/LM/3-gram.pruned.1e-7.bin
mv librispeech-vocab.txt data/LM/librispeech-vocab.txt
