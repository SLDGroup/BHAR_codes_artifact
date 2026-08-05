#!/bin/bash

# unconstrained
./unconstrained-synchronous_multisensor-pamap2.sh;

# opportunistic
./opportunistic-asynchronous_single_sensor-pamap2.sh;
./opportunistic-asynchronous_multisensor-pamap2.sh;
./opportunistic-asynchronous_multisensor_time_context-pamap2.sh;

# conservative
./conservative-asynchronous_single_sensor-pamap2.sh;
./conservative-asynchronous_multisensor-pamap2.sh;
./conservative-asynchronous_multisensor_time_context-pamap2.sh;

# sensitivity
./posttrain_pamap2-sensitivity.sh;

# heuristic
./heuristic-asynchronous_transformer-pamap2.sh