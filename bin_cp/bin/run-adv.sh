#script
#!/bin/bash

# model_sigmas=("0.12" "0.25" "0.5");
# smoothing_sigmas=("0.25" "0.5");
smoothing_sigmas=("0.25");
rs=("0.0" "0.06" "0.12" "0.18" "0.25" "0.5" "0.75")
# rs=("0.5")
score_methods=("TPS");


for rv in "${rs[@]}"; do
    for smoothing_sigma in "${smoothing_sigmas[@]}"; do
        for score_method in "${score_methods[@]}"; do
            echo "r: $rv, Smoothing sigma: $smoothing_sigma";
            python3 smooth_logits_pert.py with model_sigma=$smoothing_sigma smoothing_sigma=$smoothing_sigma r=$rv;
            python3 compare-methods-pert.py with model_sigma=$smoothing_sigma smoothing_sigma=$smoothing_sigma score_method=$score_method r=$rv;
        done
    done
done
