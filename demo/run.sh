#script
#!/bin/bash

# model_sigmas=("0.12" "0.25" "0.5");
model_sigmas=("0.5");
# smoothing_sigmas=("0.12" "0.25" "0.5");
smoothing_sigmas=("0.5");
score_methods=("TPS");
trial_samples=("70" "80" "90");


for smoothing_sigma in "${smoothing_sigmas[@]}"; do
	for score_method in "${score_methods[@]}"; do
		if [ "$smoothing_sigma" == "$model_sigma" ]; then
			echo "Model sigma and Smoothing sigma are the same. Skipping.";
			continue;
		fi
		for n_trial_samples in "${trial_samples[@]}"; do
			echo "Smoothing sigma: $smoothing_sigma, Score method: $score_method, n_trial_samples: $n_trial_samples";
			python3 l2-guass-comparison.py with model_sigma=$smoothing_sigma smoothing_sigma=$smoothing_sigma n_trial_samples=$n_trial_samples score_method=$score_method n_samples=10000;	
		done
	done
done

