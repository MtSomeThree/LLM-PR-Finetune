prefix=$1
cuda=$2
sampling_cuda=$3
let cuda2=cuda+1
thres=$4
echo "Start!"

for i in {4..6}
do
	let j=i-1
	let k=j-1
	let last=i*10000
	let start=(i+1)*10000
	let end=(i+2)*10000
	{
		python train_nado.py --cuda $cuda --num_epochs 5 --save_dir "$prefix"/NADO_"$i"_5.pt --load_dir "$prefix"/PR_"$j"_3.pt --load_dir2 "$prefix"/NADO_"$j"_5.pt --samples_file dump/"$prefix"/PR_"$k"_3_"$last"-"$start"-200-20-API.pt --threshold $thres
		python train_nado.py --cuda $cuda --eval_model nado --load_dir "$prefix"/NADO_"$i"_5.pt --save_dir "$prefix"/NADO_"$i"_5_10000-11000-200-20-None.pt
		python perspectiveAPI.py --cuda $cuda --ppl --eval --filename "$prefix"/NADO_"$i"_5_10000-11000-200-20-None.pt --batch_size 50 >> dump/"$prefix"/results.txt
	}&

	{
		python train_nado.py --cuda $cuda2 --load_dir "$prefix"/NADO_"$j"_5.pt --fine_tune --batch_size 50 --num_epochs 3 --save_dir "$prefix"/PR_"$i"_3.pt --samples_file dump/"$prefix"/PR_"$k"_3_"$last"-"$start"-200-20-API.pt
		python train_nado.py --cuda $cuda2 --eval_model gpt --load_dir "$prefix"/PR_"$i"_3.pt --save_dir "$prefix"/PR_"$i"_3_10000-11000-200-20-None.pt
		python perspectiveAPI.py --cuda $cuda2 --ppl --eval --filename "$prefix"/PR_"$i"_3_10000-11000-200-20-None.pt --batch_size 50 >> dump/"$prefix"/results.txt
	}&

	{
		python sampling.py --cuda $sampling_cuda --API --prompts_num 10000 --start_index $start --model_name PR_"$j"_3 --load_dir "$prefix"/PR_"$j"_3.pt --samples_file dump/"$prefix"/PR_"$j"_3_"$start"-"$end"-200-20-None.pt
	}&
	wait
done



