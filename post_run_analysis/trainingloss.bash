#!/bin/bash

for LAMBDA in 100000; do
	for testname in wp50_t_23_3 wp60_t_23_3 wp75_t_23_3 wp80_t_23_3 wp90_t_23_3; do
			input_path='/scratch/hannn/pix2pix-for-swot/lambda_'$LAMBDA'_nlayers_5_testing_'$testname'_700epochs_w_rotflipud/'
			python read_event_log.py \
				--input_dir $input_path 
	done
done