prefix=$1

for i in {3..3}
do
	let j=i-1
	let last=i*10000
	let start=(i+1)*10000
	let end=(i+2)*10000
	{
		python train_nado.py --cuda 7 --eval_model nado --load_dir "$prefix"/NADO_"$i"_5.pt --save_dir "$prefix"/NADO_"$i"_5_10000-11000-200-20-None.pt
		python perspectiveAPI.py --cuda 7 --ppl --eval --filename "$prefix"/NADO_"$i"_5_10000-11000-200-20-None.pt --batch_size 50 >> "$prefix"/log.txt
	}
	echo "${i}-1"
	echo "${i}-1"
	python train_nado.py --cuda 7 --load_dir "$prefix"/NADO_"$i"_5.pt --load_dir2 "$prefix"/PR_"$j"_3.pt --fine_tune --batch_size 50 --num_epochs 3 --save_dir "$prefix"/PR_"$i"_3.pt  --samples_file dump/"$prefix"/PR_"$j"_3_"$last"-"$start"-200-20-API.pt
	{
		python train_nado.py --cuda 7 --eval_model gpt --load_dir "$prefix"/PR_"$i"_3.pt --save_dir "$prefix"/PR_"$i"_3_10000-11000-200-20-None.pt
		python perspectiveAPI.py --cuda 7 --ppl --eval --filename "$prefix"/PR_"$i"_3_10000-11000-200-20-None.pt >> "$prefix"/log.txt 
	}
	echo "${i}-2"
	echo "${i}-2"
	python sampling.py --API --cuda 6 --prompts_num 10000 --start_index $start --model_name PR_"$i"_3 --load_dir "$prefix"/PR_"$i"_3.pt --samples_file dump/"$prefix"/PR_"$i"_3_"$start"-"$end"-200-20-None.pt
	echo "${i}-3"
	echo "${i}-3"
done

for i in {4..5}
do
	let j=i-1
	let last=i*10000
	let start=(i+1)*10000
	let end=(i+2)*10000
	python train_nado.py --cuda 7 --num_epochs 5 --load_dir "$prefix"/PR_"$j"_3.pt --load_dir2 "$prefix"/NADO_"$j"_5.pt --save_dir "$prefix"/NADO_"$i"_5.pt  --samples_file dump/"$prefix"/PR_"$j"_3_"$last"-"$start"-200-20-API.pt
	{
		python train_nado.py --cuda 7 --eval_model nado --load_dir "$prefix"/NADO_"$i"_5.pt --save_dir "$prefix"/NADO_"$i"_5_10000-11000-200-20-None.pt
		python perspectiveAPI.py --cuda 7 --ppl --eval --filename "$prefix"/NADO_"$i"_5_10000-11000-200-20-None.pt --batch_size 50 >> "$prefix"/log.txt
	}
	echo "${i}-1"
	echo "${i}-1"
	python train_nado.py --cuda 7 --load_dir "$prefix"/NADO_"$i"_5.pt --load_dir2 "$prefix"/PR_"$j"_3.pt --fine_tune --batch_size 50 --num_epochs 3 --save_dir "$prefix"/PR_"$i"_3.pt  --samples_file dump/"$prefix"/PR_"$j"_3_"$last"-"$start"-200-20-API.pt
	{
		python train_nado.py --cuda 7 --eval_model gpt --load_dir "$prefix"/PR_"$i"_3.pt --save_dir "$prefix"/PR_"$i"_3_10000-11000-200-20-None.pt
		python perspectiveAPI.py --cuda 7 --ppl --eval --filename "$prefix"/PR_"$i"_3_10000-11000-200-20-None.pt >> "$prefix"/log.txt 
	}
	echo "${i}-2"
	echo "${i}-2"
	python sampling.py --API --cuda 6 --prompts_num 10000 --start_index $start --model_name PR_"$i"_3 --load_dir "$prefix"/PR_"$i"_3.pt --samples_file dump/"$prefix"/PR_"$i"_3_"$start"-"$end"-200-20-None.pt
	echo "${i}-3"
	echo "${i}-3"
done

