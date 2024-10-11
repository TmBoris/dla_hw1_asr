echo 'downloading best model...'
gdown https://drive.google.com/uc?id=14kG02kXetJKLAcFgpaPPdeLFAldI5CKS
mkdir -p saved/best_model
mv model_best.pth saved/best_model/model_best.pth
