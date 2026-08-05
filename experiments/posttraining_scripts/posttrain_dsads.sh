#!/bin/bash

# unconstrained
./unconstrained-synchronous_multisensor-dsads.sh;

# opportunistic
./opportunistic-asynchronous_single_sensor-dsads.sh;
./opportunistic-asynchronous_multisensor-dsads.sh;
./opportunistic-asynchronous_multisensor_time_context-dsads.sh;

# conservative
./conservative-asynchronous_single_sensor-dsads.sh;
./conservative-asynchronous_multisensor-dsads.sh;
./conservative-asynchronous_multisensor_time_context-dsads.sh;

# heuristic
./heuristic-asynchronous_transformer-dsads.sh