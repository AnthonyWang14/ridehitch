# Ride Hitch
Use DRL method to help solve the ride-hitch problem.
## Problem Setting
There are two types of requests from the environment. 
One type is the supply type from drivers and another is the demand from passengers.


## Result
|  T50D50P100  | FIRST | RANDOM | DQN | OPT  |
|-------------------|-------|--------|-----|------|
| R100   | 21    | 23    |     | 24   |
| R1000  | 353   | 334   |     | 379  |
| R10000 | 3588  | 3323  |     | 3779 |

|  NormT50D50Pinf  | FIRST | RANDOM | DQN | OPT  |
|-------------------|-------|--------|-----|------|
| R100             | 21    | 23    |     | 24   |
| R1000            | 371   |  357  |     |  403 |
| R10000           |   |   |     |  |

|  TaxiT60D70P100  | FIRST | RANDOM | DQN | OPT  |
|-------------------|-------|--------|-----|------|
| R100   |     |     |     |     |
| R1000  |  428  |  418  |     | 428 |
| R10000 |   |   |     | |

|  TaxiT60D70P1000  | FIRST | RANDOM | DQN | OPT  |
|-------------------|-------|--------|-----|------|
| R100   |     |     |     |     |
| R1000  |  429  | 423 |     | 439 |
| R10000 |   |   |     | |




