#!/bin/bash

# unconstrained
./unconstrained-synchronous_multisensor-rwhar.sh;

# opportunistic
./opportunistic-asynchronous_single_sensor-rwhar.sh;
./opportunistic-asynchronous_multisensor-rwhar.sh;
./opportunistic-asynchronous_multisensor_time_context-rwhar.sh;

# conservative
./conservative-asynchronous_single_sensor-rwhar.sh;
./conservative-asynchronous_multisensor-rwhar.sh;
./conservative-asynchronous_multisensor_time_context-rwhar.sh;