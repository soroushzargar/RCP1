#script
#!/bin/bash

model_sigmas=("0.12" "0.25" "0.5");
smoothing_sigmas=("0.12" "0.25" "0.5");
score_methods=("TPS");


for model_sigma in "${model_sigmas[@]}"; do
	for smoothing_sigma in "${smoothing_sigmas[@]}"; do
       		for score_method in "${score_methods[@]}"; do
				if [ "$smoothing_sigma" == "$model_sigma" ]; then
					echo "Model sigma and Smoothing sigma are the same. Skipping.";
					continue;
				fi
				echo "Model sigma: $model_sigma, Smoothing sigma: $smoothing_sigma";
				python3 smooth_logits_clean.py with model_sigma=$model_sigma smoothing_sigma=$smoothing_sigma;
				python3 compare-methods-clean.py with model_sigma=$model_sigma smoothing_sigma=$smoothing_sigma score_method=$score_method n_trial_samples=1000;
				python3 compare-methods-clean.py with model_sigma=$model_sigma smoothing_sigma=$smoothing_sigma score_method=$score_method n_trial_samples=2000;
				#python3 compare-methods-clean.py with model_sigma=$model_sigma smoothing_sigma=$smoothing_sigma score_method=$score_method n_trial_samples=5000;
				python3 compare-methods-clean.py with model_sigma=$model_sigma smoothing_sigma=$smoothing_sigma score_method=$score_method n_trial_samples=10000;
       		done
       	done
done

