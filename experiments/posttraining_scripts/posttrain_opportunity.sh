#!/bin/bash

# unconstrained
./unconstrained-synchronous_multisensor-opportunity.sh;

# opportunistic
./opportunistic-asynchronous_single_sensor-opportunity.sh;
./opportunistic-asynchronous_multisensor-opportunity.sh;
./opportunistic-asynchronous_multisensor_time_context-opportunity.sh;

# conservative
./conservative-asynchronous_single_sensor-opportunity.sh;
./conservative-asynchronous_multisensor-opportunity.sh;
./conservative-asynchronous_multisensor_time_context-opportunity.sh;