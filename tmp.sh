python train_nado.py --cuda 3 --eval_model base --load_dir dump/base/base_0_5.pt --save_dir dump/base/base_10000-11000-200-20-None.pt
python perspectiveAPI.py --cuda 3 --ppl --eval --filename dump/base/base_10000-11000-200-20-None.pt
