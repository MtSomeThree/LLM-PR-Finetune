prefix=$1
for name in dump/"$prefix"/NADO*_5.pt
do 
	file=${name/.pt/_10000-11000-200-20-API.pt}
	echo $file
	#python train_nado.py --cuda 7 --eval_model nado --load_dir $name --save_dir $file
	python perspectiveAPI.py --cuda 3 --ppl --eval --filename $file >> dump/"$prefix"/results200.txt 
	python perspectiveAPI.py --cuda 3 --eval --bin_size 25 --filename $file  >> dump/"$prefix"/results25.txt
done

for name in dump/"$prefix"/PR*_3.pt
do 
	file=${name/.pt/_10000-11000-200-20-API.pt}
	echo $file
	#python train_nado.py --cuda 7 --eval_model gpt --load_dir $name --save_dir $file
	python perspectiveAPI.py --cuda 3 --ppl --eval --filename $file >> dump/"$prefix"/results200.txt
	python perspectiveAPI.py --cuda 3 --eval --bin_size 25 --filename $file >> dump/"$prefix"/results25.txt 
done